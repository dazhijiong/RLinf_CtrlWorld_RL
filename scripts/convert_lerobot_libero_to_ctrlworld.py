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

"""Convert LeRobot LIBERO image dataset to the Ctrl-World training layout.

The output layout mirrors the subset used by the original Ctrl-World code:

- annotation/{train,val}/{episode_id}.json
- videos/{train,val}/{episode_id}/{0,1,2}.mp4
- latent_videos/{train,val}/{episode_id}/{0,1,2}.pt

Notes:
- The source LeRobot LIBERO dataset only provides one third-person view and one
  wrist view. This script therefore writes:
  - view 0 = main
  - view 1 = main (duplicated)
  - view 2 = wrist
- Ctrl-World's dataset loader assumes state streams are 3x denser than the
  stored video stream. To stay compatible without changing their loader, this
  script repeats each retained state `state_repeat` times.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import mediapy
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLTemporalDecoder

from rlinf.data.datasets.lerobot_world_model import LeRobotTrajectoryDatasetWrapper


class CtrlWorldLeRobotConverter(LeRobotTrajectoryDatasetWrapper):
    def __init__(
        self,
        data_dir: str,
        camera_heights: int,
        camera_widths: int,
        state_repeat: int,
        gripper_strategy: str,
    ):
        super().__init__(
            data_dir=data_dir,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
        )
        self.state_repeat = int(state_repeat)
        if self.state_repeat < 1:
            raise ValueError(f"state_repeat must be >= 1, got {state_repeat}")
        self.gripper_strategy = gripper_strategy

    def _get_episode_rows(self, episode: Dict[str, Any]) -> List[Dict[str, Any]]:
        episode_index = int(episode["episode_index"])
        data_file = self._get_data_file_for_episode(episode)
        table = pq.read_table(
            data_file,
            columns=[
                "observation.images.image",
                "observation.images.wrist_image",
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
            ],
            filters=[("episode_index", "=", episode_index)],
        )
        rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
        if not rows:
            raise ValueError(f"No rows found for episode {episode_index} in {data_file}")
        return rows

    def _gripper_from_state(self, state: np.ndarray) -> float:
        if state.shape[0] < 7:
            raise ValueError(f"Expected state dim >= 7, got {state.shape[0]}")
        if state.shape[0] == 7:
            return float(state[6])
        gripper_state = state[6:]
        if self.gripper_strategy == "first":
            return float(gripper_state[0])
        if self.gripper_strategy == "mean":
            return float(gripper_state.mean())
        if self.gripper_strategy == "mean_abs":
            return float(np.abs(gripper_state).mean())
        raise ValueError(f"Unsupported gripper_strategy: {self.gripper_strategy}")

    def state_to_ctrlworld_pose(self, state: List[float]) -> List[float]:
        state_np = np.asarray(state, dtype=np.float32)
        pose = np.concatenate([state_np[:6], np.array([self._gripper_from_state(state_np)])])
        return pose.astype(np.float32).tolist()

    def action_to_ctrlworld_action(self, action: List[float]) -> List[float]:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.shape[0] < 7:
            raise ValueError(f"Expected action dim >= 7, got {action_np.shape[0]}")
        return action_np[:7].astype(np.float32).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=str,
        required=True,
        help="Root of the local LeRobot dataset (e.g. lerobot_libero_spatial_image).",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Output root using Ctrl-World dataset layout.",
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        required=True,
        help="Stable Video Diffusion model path containing the VAE subfolder.",
    )
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every N-th frame from the LeRobot episode.",
    )
    parser.add_argument(
        "--state-repeat",
        type=int,
        default=3,
        help="Repeat each retained state this many times for Ctrl-World compatibility.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS written into the exported mp4 files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of episodes written to val split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--gripper-strategy",
        type=str,
        default="mean_abs",
        choices=["first", "mean", "mean_abs"],
        help="How to turn LIBERO's multi-dim gripper state into Ctrl-World's 1-dim gripper.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode outputs.",
    )
    return parser.parse_args()


def split_episodes(
    episodes: List[Dict[str, Any]], val_ratio: float, seed: int
) -> Dict[int, str]:
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
    mediapy.write_video(os.fspath(video_path), video, fps=fps)


def build_annotation(
    episode_id: int,
    instruction: str,
    split: str,
    video_length: int,
    states: List[List[float]],
    raw_cartesian_position: List[List[float]],
    raw_gripper_position: List[float],
    raw_actions: List[List[float]],
) -> Dict[str, Any]:
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


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    device = torch.device(args.device)

    converter = CtrlWorldLeRobotConverter(
        data_dir=os.fspath(src_root),
        camera_heights=args.image_height,
        camera_widths=args.image_width,
        state_repeat=args.state_repeat,
        gripper_strategy=args.gripper_strategy,
    )
    split_map = split_episodes(converter.episodes, args.val_ratio, args.seed)

    ensure_dir(dst_root)
    for split in ("train", "val"):
        ensure_dir(dst_root / "annotation" / split)
        ensure_dir(dst_root / "videos" / split)
        ensure_dir(dst_root / "latent_videos" / split)

    manifest: List[Dict[str, Any]] = []

    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae").to(device)
    vae.eval()

    episodes = converter.episodes
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    for episode in episodes:
        episode_id = int(episode["episode_index"])
        split = split_map[episode_id]
        instruction = str(episode.get("tasks", [""])[0])

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

        main_frames = []
        wrist_frames = []
        state_per_frame = []
        action_per_frame = []
        for row in rows:
            main_frames.append(converter._decode_image(row["observation.images.image"]))
            wrist_frames.append(converter._decode_image(row["observation.images.wrist_image"]))
            state_per_frame.append(converter.state_to_ctrlworld_pose(row["observation.state"]))
            action_per_frame.append(converter.action_to_ctrlworld_action(row["action"]))

        main_video = torch.stack(main_frames, dim=0)
        wrist_video = torch.stack(wrist_frames, dim=0)
        duplicated_main_video = main_video.clone()

        for view_id, frames in enumerate((main_video, duplicated_main_video, wrist_video)):
            write_mp4(video_dir / f"{view_id}.mp4", frames, fps=args.fps)
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
        "frame_stride": args.frame_stride,
        "state_repeat": args.state_repeat,
        "fps": args.fps,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "gripper_strategy": args.gripper_strategy,
        "episodes_total": len(episodes),
    }
    with open(dst_root / "conversion_config.json", "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)


if __name__ == "__main__":
    main()
