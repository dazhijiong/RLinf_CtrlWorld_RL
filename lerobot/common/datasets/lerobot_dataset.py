"""Minimal local LeRobot dataset implementation for OpenPI SFT.

This shim is intentionally small: it only implements the parts used by
``openpi.training.data_loader`` inside RLinf.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def _resolve_dataset_root(repo_id: str) -> Path:
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
        f"Could not resolve local LeRobot dataset for repo_id={repo_id!r}. "
        f"Tried: {tried}"
    )


def _decode_image(image_struct: dict[str, Any], dataset_root: Path) -> np.ndarray:
    image_bytes = image_struct.get("bytes")
    image_path = image_struct.get("path")
    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif image_path:
        image = Image.open(dataset_root / image_path).convert("RGB")
    else:
        raise ValueError("LeRobot image struct contains neither bytes nor path.")
    return np.asarray(image, dtype=np.uint8)


@dataclass(frozen=True)
class LeRobotDatasetMetadata:
    repo_id: str

    def __post_init__(self) -> None:
        root = _resolve_dataset_root(self.repo_id)
        info = json.loads((root / "meta" / "info.json").read_text())
        tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
        tasks = {
            int(row["task_index"]): str(row["task"])
            for row in tasks_table.to_pylist()
        }
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "fps", int(info["fps"]))
        object.__setattr__(self, "tasks", tasks)


class LeRobotDataset:
    """Small random-access dataset returning one sample per frame window."""

    def __init__(
        self,
        repo_id: str,
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.root = _resolve_dataset_root(repo_id)
        self.meta = LeRobotDatasetMetadata(repo_id)
        self.delta_timestamps = delta_timestamps or {}

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
            valid_len = max(0, length - self.action_horizon + 1)
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
                "observation.images.image",
                "observation.images.wrist_image",
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
        action_rows = rows[frame_index : frame_index + self.action_horizon]

        return {
            "observation.images.image": _decode_image(
                row["observation.images.image"], self.root
            ),
            "observation.images.wrist_image": _decode_image(
                row["observation.images.wrist_image"], self.root
            ),
            "observation.state": np.asarray(
                row["observation.state"], dtype=np.float32
            ),
            "action": np.asarray(
                [action_row["action"] for action_row in action_rows], dtype=np.float32
            ),
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
        }
