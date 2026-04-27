# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for the local LeRobot-format book dataset."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

BOOK_MAIN_IMAGE_KEY = "observation.images.d405_rgb"
BOOK_EXTRA_VIEW_IMAGE_KEY = "observation.images.d405_1_rgb"
BOOK_WRIST_IMAGE_KEY = "observation.images.d435_rgb"
BOOK_IMAGE_KEYS = (
    BOOK_MAIN_IMAGE_KEY,
    BOOK_EXTRA_VIEW_IMAGE_KEY,
    BOOK_WRIST_IMAGE_KEY,
)


def resolve_lerobot_dataset_root(repo_id: str) -> Path:
    """Resolve a local LeRobot dataset root from HF_LEROBOT_HOME and repo_id."""
    env_root = os.environ.get("HF_LEROBOT_HOME")
    candidates: list[Path] = []
    if env_root:
        base = Path(env_root).expanduser().resolve()
        candidates.append(base)
        candidates.append(base / repo_id)
    candidates.append(Path(repo_id).expanduser())

    for candidate in candidates:
        if (candidate / "meta").is_dir() and (candidate / "data").is_dir():
            return candidate.resolve()

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve local LeRobot dataset for repo_id={repo_id!r}. Tried: {tried}"
    )


def load_lerobot_task_map(root: Path) -> dict[int, str]:
    tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
    tasks: dict[int, str] = {}
    for row in tasks_table.to_pylist():
        task = row.get("task", row.get("__index_level_0__"))
        if task is None:
            raise KeyError(
                "Expected 'task' or '__index_level_0__' in tasks.parquet rows."
            )
        tasks[int(row["task_index"])] = str(task)
    return tasks


def load_book_image_frame(
    root: Path, episode_index: int, image_key: str, frame_index: int
) -> np.ndarray:
    image_path = (
        root
        / "images"
        / image_key
        / f"episode-{episode_index:06d}"
        / f"frame-{frame_index:06d}.png"
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing frame for {image_key}: {image_path}")
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


@dataclass(frozen=True)
class LeRobotBookDatasetMetadata:
    repo_id: str

    def __post_init__(self) -> None:
        root = resolve_lerobot_dataset_root(self.repo_id)
        info = json.loads((root / "meta" / "info.json").read_text())

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "fps", int(info["fps"]))
        object.__setattr__(self, "tasks", load_lerobot_task_map(root))


class LeRobotBookDataset:
    """Random-access dataset for OpenPI SFT on the local book dataset."""

    def __init__(
        self,
        repo_id: str,
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
        frame_stride: int = 1,
    ) -> None:
        self.repo_id = repo_id
        self.root = resolve_lerobot_dataset_root(repo_id)
        self.meta = LeRobotBookDatasetMetadata(repo_id)
        self.delta_timestamps = delta_timestamps or {}
        self.frame_stride = max(1, int(frame_stride))
        self.info = self.meta.info

        self.image_keys = [
            key
            for key, feature in self.info["features"].items()
            if key.startswith("observation.images.") and feature.get("dtype") == "video"
        ]
        if set(self.image_keys) != set(BOOK_IMAGE_KEYS):
            raise ValueError(
                f"Expected book image keys {BOOK_IMAGE_KEYS}, got {self.image_keys}"
            )

        action_key = next(iter(self.delta_timestamps), "action")
        action_offsets = self.delta_timestamps.get(action_key, [0.0])
        self.action_horizon = max(1, len(action_offsets))

        episodes_path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_table = pq.read_table(episodes_path)
        self.episodes = sorted(
            episodes_table.to_pylist(), key=lambda row: int(row["episode_index"])
        )

        self._episode_rows: dict[int, list[dict[str, Any]]] = {}
        self._samples: list[tuple[int, int]] = []
        for episode in self.episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            required_frames = 1 + (self.action_horizon - 1) * self.frame_stride
            valid_len = max(0, length - required_frames + 1)
            for frame_idx in range(valid_len):
                self._samples.append((episode_index, frame_idx))

    def __len__(self) -> int:
        return len(self._samples)

    def _get_data_file_for_episode(self, episode_index: int) -> Path:
        episode = self.episodes[episode_index]
        chunk_index = int(episode["data/chunk_index"])
        file_index = int(episode["data/file_index"])
        return self.root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def _load_episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        if episode_index in self._episode_rows:
            return self._episode_rows[episode_index]

        data_file = self._get_data_file_for_episode(episode_index)
        table = pq.read_table(
            data_file,
            columns=[
                "observation.state",
                "action",
                "task_index",
                "episode_index",
                "frame_index",
            ],
            filters=[("episode_index", "=", episode_index)],
        )
        rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
        self._episode_rows[episode_index] = rows
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self._samples[index]
        rows = self._load_episode_rows(episode_index)

        row = rows[frame_index]
        action_indices = range(
            frame_index,
            frame_index + self.action_horizon * self.frame_stride,
            self.frame_stride,
        )
        action_rows = [rows[action_idx] for action_idx in action_indices]
        row_frame_index = int(row["frame_index"])

        sample = {
            "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
            "action": np.asarray(
                [action_row["action"] for action_row in action_rows], dtype=np.float32
            ),
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
        }
        for image_key in self.image_keys:
            sample[image_key] = load_book_image_frame(
                self.root, episode_index, image_key, row_frame_index
            )
        return sample


class LeRobotBookTrajectoryDatasetWrapper(Dataset):
    """Read reset states from the local book dataset for world-model rollout."""

    def __init__(
        self,
        data_dir: str,
        camera_heights: Optional[int] = None,
        camera_widths: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.meta_dir = self.data_dir / "meta"
        self.data_files_dir = self.data_dir / "data"
        if not self.meta_dir.is_dir() or not self.data_files_dir.is_dir():
            raise ValueError(
                f"LeRobot dataset root must contain meta/ and data/: {self.data_dir}"
            )

        episodes_meta_path = self.meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
        if not episodes_meta_path.exists():
            raise ValueError(f"Missing LeRobot episode metadata: {episodes_meta_path}")

        self.image_transforms = None
        if camera_heights is not None and camera_widths is not None:
            self.image_transforms = transforms.Compose(
                [transforms.Resize((camera_heights, camera_widths))]
            )

        episodes_table = pq.read_table(episodes_meta_path)
        self.episodes = sorted(
            episodes_table.to_pylist(), key=lambda row: int(row["episode_index"])
        )
        self.task_map = load_lerobot_task_map(self.data_dir)
        self._episode_cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.episodes)

    def _get_data_file_for_episode(self, episode: dict[str, Any]) -> Path:
        chunk_index = int(episode["data/chunk_index"])
        file_index = int(episode["data/file_index"])
        return self.data_dir / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def _get_first_frame(self, episode: dict[str, Any]) -> dict[str, Any]:
        episode_index = int(episode["episode_index"])
        if episode_index in self._episode_cache:
            return self._episode_cache[episode_index]

        data_file = self._get_data_file_for_episode(episode)
        first_frame_table = pq.read_table(
            data_file,
            columns=[
                "observation.state",
                "episode_index",
                "frame_index",
                "task_index",
            ],
            filters=[
                ("episode_index", "=", episode_index),
                ("frame_index", "=", 0),
            ],
        )
        rows = first_frame_table.to_pylist()
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one first frame for episode {episode_index}, got {len(rows)}"
            )
        self._episode_cache[episode_index] = rows[0]
        return rows[0]

    def _load_tensor_image(
        self, episode_index: int, image_key: str, frame_index: int
    ) -> torch.Tensor:
        image_np = load_book_image_frame(self.data_dir, episode_index, image_key, frame_index)
        image_tensor = transforms.ToTensor()(image_np)
        if self.image_transforms is not None:
            image_tensor = self.image_transforms(image_tensor)
        return image_tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        episode_index = int(episode["episode_index"])
        first_frame = self._get_first_frame(episode)
        frame_index = int(first_frame["frame_index"])

        main_image = self._load_tensor_image(episode_index, BOOK_MAIN_IMAGE_KEY, frame_index)
        extra_view_image = self._load_tensor_image(
            episode_index, BOOK_EXTRA_VIEW_IMAGE_KEY, frame_index
        )
        wrist_image = self._load_tensor_image(
            episode_index, BOOK_WRIST_IMAGE_KEY, frame_index
        )
        state = torch.tensor(first_frame["observation.state"], dtype=torch.float32)

        tasks = episode.get("tasks", [])
        if tasks:
            task = str(tasks[0])
        else:
            task = self.task_map.get(int(first_frame["task_index"]), "")

        start_frame = {
            "image": main_image,
            "main_image_1": main_image,
            "main_image_2": extra_view_image,
            "wrist_image": wrist_image,
            "observation.state": state,
        }

        return {
            "start_items": [start_frame],
            "target_items": [],
            "episode_index": episode_index,
            "task": task,
            "dataset_meta": {
                "episode_length": int(episode.get("length", 1)),
                "file_path": os.fspath(self.data_dir),
                "task_index": int(first_frame["task_index"]),
            },
        }
