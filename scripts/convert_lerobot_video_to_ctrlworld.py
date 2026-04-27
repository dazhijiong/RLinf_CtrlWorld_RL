#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a video-backed local LeRobot dataset to Ctrl-World training layout.

This variant handles datasets where:
- states/actions live in parquet files under ``data/``
- videos are stored as chunked mp4 files under ``videos/``
- episode metadata in ``meta/episodes/...`` provides per-episode time ranges

The output mirrors the layout expected by Ctrl-World:

- annotation/{train,val}/{episode_id}.json
- videos/{train,val}/{episode_id}/{0,1,2}.mp4
- latent_videos/{train,val}/{episode_id}/{0,1,2}.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLTemporalDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, required=True, help="Root of the local LeRobot dataset.")
    parser.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Output root using the Ctrl-World dataset layout.",
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        required=True,
        help="Stable Video Diffusion model path containing the VAE subfolder.",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        nargs="+",
        default=None,
        help="Three video feature keys to export as Ctrl-World view 0/1/2.",
    )
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every N-th state/video frame from each episode.",
    )
    parser.add_argument(
        "--state-repeat",
        type=int,
        default=3,
        help="Repeat each retained state this many times for Ctrl-World compatibility.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS. Defaults to source FPS.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_episodes(episodes: list[dict[str, Any]], val_ratio: float, seed: int) -> dict[int, str]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    episode_ids = [int(ep["episode_index"]) for ep in episodes]
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val_ids = set(shuffled[:val_count])
    return {episode_id: ("val" if episode_id in val_ids else "train") for episode_id in episode_ids}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def encode_video_to_latents(
    vae: AutoencoderKLTemporalDecoder,
    frames_01: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    frames = frames_01.to(device=device, dtype=torch.float32)
    frames = frames * 2.0 - 1.0
    latents = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start : start + batch_size]
            latent = vae.encode(batch).latent_dist.sample()
            latent = latent.mul_(vae.config.scaling_factor).cpu()
            latents.append(latent)
    return torch.cat(latents, dim=0)


def write_mp4(video_path: Path, frames_01: torch.Tensor, fps: float) -> None:
    frames_uint8 = (frames_01.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    video = frames_uint8.permute(0, 2, 3, 1).cpu().numpy()
    writer = imageio.get_writer(os.fspath(video_path), format="ffmpeg", fps=fps)
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def build_annotation(
    episode_id: int,
    instruction: str,
    split: str,
    video_length: int,
    states: list[list[float]],
    raw_cartesian_position: list[list[float]],
    raw_gripper_position: list[float],
    raw_actions: list[list[float]],
) -> dict[str, Any]:
    zero_joint = [[0.0] * 7 for _ in range(len(raw_cartesian_position))]
    return {
        "texts": [instruction],
        "episode_id": episode_id,
        "success": 1,
        "video_length": video_length,
        "state_length": len(states),
        "raw_length": len(raw_cartesian_position),
        "videos": [
            {"video_path": f"videos/{split}/{episode_id}/0.mp4"},
            {"video_path": f"videos/{split}/{episode_id}/1.mp4"},
            {"video_path": f"videos/{split}/{episode_id}/2.mp4"},
        ],
        "latent_videos": [
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/0.pt"},
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/1.pt"},
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/2.pt"},
        ],
        "states": states,
        "observation.state.cartesian_position": raw_cartesian_position,
        "observation.state.joint_position": zero_joint,
        "observation.state.gripper_position": raw_gripper_position,
        "action.cartesian_position": [action[:6] for action in raw_actions],
        "action.joint_position": zero_joint,
        "action.gripper_position": [action[6] for action in raw_actions],
        "action.joint_velocity": zero_joint,
    }


class VideoBackedLeRobotConverter:
    def __init__(
        self,
        data_dir: str,
        camera_heights: int,
        camera_widths: int,
        camera_keys: list[str] | None,
        state_repeat: int,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.state_repeat = int(state_repeat)
        if self.state_repeat < 1:
            raise ValueError(f"state_repeat must be >= 1, got {state_repeat}")
        self.camera_heights = int(camera_heights)
        self.camera_widths = int(camera_widths)

        info_path = self.data_dir / "meta" / "info.json"
        if not info_path.exists():
            raise ValueError(f"Missing LeRobot dataset metadata: {info_path}")
        with open(info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

        available_video_keys = [
            key
            for key, value in self.info["features"].items()
            if isinstance(value, dict) and value.get("dtype") == "video"
        ]
        if camera_keys is None:
            camera_keys = available_video_keys[:3]
        if len(camera_keys) != 3:
            raise ValueError(f"Expected exactly 3 camera keys, got {camera_keys}")
        missing_keys = [key for key in camera_keys if key not in available_video_keys]
        if missing_keys:
            raise ValueError(
                f"Camera keys {missing_keys} not found in dataset. Available: {available_video_keys}"
            )
        self.camera_keys = list(camera_keys)
        self.video_fps = float(self.info["fps"])

        episodes_meta_path = self.data_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not episodes_meta_path.exists():
            raise ValueError(f"Missing episode metadata: {episodes_meta_path}")
        episodes_table = pq.read_table(episodes_meta_path)
        self.episodes = sorted(episodes_table.to_pylist(), key=lambda row: row["episode_index"])

        self._video_reader_cache: dict[Path, tuple[Any, int]] = {}

    def _get_data_file_for_episode(self, episode: dict[str, Any]) -> Path:
        chunk_index = int(episode["data/chunk_index"])
        file_index = int(episode["data/file_index"])
        return self.data_dir / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def _get_video_path_for_episode(self, episode: dict[str, Any], video_key: str) -> Path:
        chunk_index = int(episode[f"videos/{video_key}/chunk_index"])
        file_index = int(episode[f"videos/{video_key}/file_index"])
        rel_path = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        return self.data_dir / rel_path

    def _get_video_reader(self, video_path: Path) -> tuple[Any, int]:
        if video_path not in self._video_reader_cache:
            reader = imageio.get_reader(os.fspath(video_path), format="ffmpeg")
            self._video_reader_cache[video_path] = (reader, reader.count_frames())
        return self._video_reader_cache[video_path]

    def _get_episode_rows(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        episode_index = int(episode["episode_index"])
        data_file = self._get_data_file_for_episode(episode)
        table = pq.read_table(
            data_file,
            columns=["action", "observation.state", "timestamp", "frame_index", "episode_index"],
            filters=[("episode_index", "=", episode_index)],
        )
        rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
        if not rows:
            raise ValueError(f"No rows found for episode {episode_index} in {data_file}")
        return rows

    def _resize_frames(self, frames_uint8: np.ndarray) -> torch.Tensor:
        frames = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
        if frames.shape[-2:] != (self.camera_heights, self.camera_widths):
            frames = F.interpolate(
                frames,
                size=(self.camera_heights, self.camera_widths),
                mode="bilinear",
                align_corners=False,
            )
        return frames

    def load_video_frames(
        self,
        episode: dict[str, Any],
        video_key: str,
        local_frame_indices: list[int],
    ) -> torch.Tensor:
        video_path = self._get_video_path_for_episode(episode, video_key)
        video_reader, num_frames = self._get_video_reader(video_path)
        start_timestamp = float(episode[f"videos/{video_key}/from_timestamp"])
        start_frame = int(round(start_timestamp * self.video_fps))
        global_frame_indices = [
            min(max(start_frame + int(frame_index), 0), num_frames - 1)
            for frame_index in local_frame_indices
        ]
        frames = np.stack([video_reader.get_data(frame_index) for frame_index in global_frame_indices], axis=0)
        return self._resize_frames(frames)

    @staticmethod
    def state_to_ctrlworld_pose(state: list[float]) -> list[float]:
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.shape[0] < 7:
            raise ValueError(f"Expected state dim >= 7, got {state_np.shape[0]}")
        return state_np[:7].astype(np.float32).tolist()

    @staticmethod
    def action_to_ctrlworld_action(action: list[float]) -> list[float]:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.shape[0] < 7:
            raise ValueError(f"Expected action dim >= 7, got {action_np.shape[0]}")
        return action_np[:7].astype(np.float32).tolist()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    device = torch.device(args.device)

    converter = VideoBackedLeRobotConverter(
        data_dir=os.fspath(src_root),
        camera_heights=args.image_height,
        camera_widths=args.image_width,
        camera_keys=args.camera_keys,
        state_repeat=args.state_repeat,
    )
    split_map = split_episodes(converter.episodes, args.val_ratio, args.seed)

    ensure_dir(dst_root)
    for split in ("train", "val"):
        ensure_dir(dst_root / "annotation" / split)
        ensure_dir(dst_root / "videos" / split)
        ensure_dir(dst_root / "latent_videos" / split)

    manifest: list[dict[str, Any]] = []

    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae").to(device)
    vae.eval()

    episodes = converter.episodes
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    output_fps = float(args.fps) if args.fps is not None else converter.video_fps

    for episode in episodes:
        episode_id = int(episode["episode_index"])
        split = split_map[episode_id]
        tasks = episode.get("tasks", [])
        instruction = str(tasks[0]) if tasks else ""

        annotation_path = dst_root / "annotation" / split / f"{episode_id}.json"
        video_dir = dst_root / "videos" / split / str(episode_id)
        latent_dir = dst_root / "latent_videos" / split / str(episode_id)
        if annotation_path.exists() and video_dir.is_dir() and latent_dir.is_dir() and not args.overwrite:
            manifest.append(
                {
                    "episode_id": episode_id,
                    "split": split,
                    "instruction": instruction,
                    "status": "skipped_existing",
                }
            )
            continue

        ensure_dir(video_dir)
        ensure_dir(latent_dir)

        rows = converter._get_episode_rows(episode)
        rows = rows[:: args.frame_stride]
        if not rows:
            raise ValueError(f"Episode {episode_id} becomes empty after frame-stride={args.frame_stride}")

        local_frame_indices = [int(row["frame_index"]) for row in rows]
        state_per_frame = [
            converter.state_to_ctrlworld_pose(row["observation.state"])
            for row in rows
        ]
        action_per_frame = [
            converter.action_to_ctrlworld_action(row["action"])
            for row in rows
        ]

        for view_id, camera_key in enumerate(converter.camera_keys):
            frames = converter.load_video_frames(episode, camera_key, local_frame_indices)
            write_mp4(video_dir / f"{view_id}.mp4", frames, fps=output_fps)
            latents = encode_video_to_latents(
                vae=vae,
                frames_01=frames,
                batch_size=args.batch_size,
                device=device,
            )
            torch.save(latents, latent_dir / f"{view_id}.pt")

        raw_cartesian_position = []
        raw_gripper_position = []
        raw_actions = []
        for state, action in zip(state_per_frame, action_per_frame):
            for _ in range(args.state_repeat):
                raw_cartesian_position.append(state[:6])
                raw_gripper_position.append(state[6])
                raw_actions.append(action)

        annotation = build_annotation(
            episode_id=episode_id,
            instruction=instruction,
            split=split,
            video_length=len(state_per_frame),
            states=state_per_frame,
            raw_cartesian_position=raw_cartesian_position,
            raw_gripper_position=raw_gripper_position,
            raw_actions=raw_actions,
        )
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

        manifest.append(
            {
                "episode_id": episode_id,
                "split": split,
                "instruction": instruction,
                "video_length": len(state_per_frame),
                "raw_length": len(raw_cartesian_position),
                "camera_keys": converter.camera_keys,
                "status": "converted",
            }
        )
        print(
            f"[convert] episode={episode_id} split={split} "
            f"frames={len(state_per_frame)} raw={len(raw_cartesian_position)}"
        )

    with open(dst_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    config_summary = {
        "src_root": os.fspath(src_root),
        "dst_root": os.fspath(dst_root),
        "svd_path": os.fspath(Path(args.svd_path).expanduser().resolve()),
        "image_size": [args.image_height, args.image_width],
        "camera_keys": converter.camera_keys,
        "source_fps": converter.video_fps,
        "output_fps": output_fps,
        "frame_stride": args.frame_stride,
        "state_repeat": args.state_repeat,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "episodes_total": len(episodes),
    }
    with open(dst_root / "conversion_config.json", "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)


if __name__ == "__main__":
    main()
