#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import pickle
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one raw GELLO episode directory into one RLinf replay-buffer rank directory."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-rank-dir", type=Path, required=True)
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    pil = pil.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _sorted_frame_paths(source_dir: Path) -> list[Path]:
    frame_paths = sorted(source_dir.glob("*.pkl"))
    if not frame_paths:
        raise FileNotFoundError(f"No .pkl frames found under {source_dir}")
    return frame_paths


def _load_frames(
    source_dir: Path, image_height: int, image_width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_paths = _sorted_frame_paths(source_dir)
    states = []
    main_images = []
    wrist_images = []
    extra_view_images = []

    for frame_path in frame_paths:
        frame = _load_pkl(frame_path)
        states.append(np.asarray(frame["ee_pos_quat"], dtype=np.float32))
        main_images.append(
            _resize_rgb(frame["RealsenseD405-0_rgb"], image_height, image_width)
        )
        wrist_images.append(
            _resize_rgb(frame["RealsenseD435_rgb"], image_height, image_width)
        )
        extra_view_images.append(
            _resize_rgb(frame["RealsenseD405-1_rgb"], image_height, image_width)
        )

    return (
        np.stack(states, axis=0),
        np.stack(main_images, axis=0),
        np.stack(wrist_images, axis=0),
        np.stack(extra_view_images, axis=0),
    )


def _build_trajectory_payload(
    *,
    source_dir: Path,
    states: np.ndarray,
    main_images: np.ndarray,
    wrist_images: np.ndarray,
    extra_view_images: np.ndarray,
    chunk_size: int,
) -> dict:
    if states.shape[0] < chunk_size:
        raise ValueError(
            f"Need at least {chunk_size} frames to build one chunk, got {states.shape[0]}"
        )

    num_samples = states.shape[0] - chunk_size + 1
    action_dim = states.shape[1]
    if action_dim != 7:
        raise ValueError(f"Expected ee_pos_quat to have 7 dims, got {action_dim}")

    actions = np.stack(
        [states[t : t + chunk_size].reshape(-1) for t in range(num_samples)], axis=0
    ).astype(np.float32)
    obs_states = states[:num_samples].astype(np.float32)
    obs_main = main_images[:num_samples].astype(np.uint8)
    obs_wrist = wrist_images[:num_samples].astype(np.uint8)
    obs_extra = extra_view_images[:num_samples].astype(np.uint8)

    model_weights_id = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, str(source_dir.expanduser().resolve()))
    )

    rewards = np.zeros((num_samples, 1, chunk_size), dtype=np.float32)
    dones = np.zeros((num_samples, 1, chunk_size), dtype=bool)
    terminations = np.zeros((num_samples, 1, chunk_size), dtype=bool)
    truncations = np.zeros((num_samples, 1, chunk_size), dtype=bool)
    dones[-1, 0, -1] = True
    terminations[-1, 0, -1] = True

    payload = {
        "max_episode_length": int(states.shape[0]),
        "model_weights_id": model_weights_id,
        "actions": torch.from_numpy(actions[:, None, :]),
        "intervene_flags": torch.zeros(
            (num_samples, 1, chunk_size * action_dim), dtype=torch.bool
        ),
        "rewards": torch.from_numpy(rewards),
        "terminations": torch.from_numpy(terminations),
        "truncations": torch.from_numpy(truncations),
        "dones": torch.from_numpy(dones),
        "versions": torch.zeros((num_samples, 1, chunk_size, action_dim), dtype=torch.float32),
        "forward_inputs": {
            "chains": torch.zeros((num_samples, 1, 4, 10, 32), dtype=torch.float32),
            "denoise_inds": torch.zeros((num_samples, 1, 3), dtype=torch.int64),
            "tokenized_prompt": torch.zeros((num_samples, 1, 200), dtype=torch.int64),
            "tokenized_prompt_mask": torch.zeros(
                (num_samples, 1, 200), dtype=torch.bool
            ),
            "action": torch.from_numpy(actions[:, None, :]),
            "model_action": torch.zeros((num_samples, 1, 320), dtype=torch.float32),
            "observation/image": torch.from_numpy(obs_main[:, None, ...]),
            "observation/state": torch.from_numpy(obs_states[:, None, :]),
            "observation/wrist_image": torch.from_numpy(obs_wrist[:, None, ...]),
            "observation/extra_view_image": torch.from_numpy(obs_extra[:, None, ...]),
        },
        "curr_obs": {},
        "next_obs": {},
    }
    return payload


def _write_rank_dir(
    *,
    output_rank_dir: Path,
    trajectory_id: int,
    payload: dict,
    num_samples: int,
    chunk_size: int,
    seed: int,
) -> None:
    output_rank_dir.mkdir(parents=True, exist_ok=True)
    model_weights_id = payload["model_weights_id"]
    traj_name = f"trajectory_{trajectory_id}_{model_weights_id}.pt"
    traj_path = output_rank_dir / traj_name
    torch.save(payload, traj_path)

    metadata = {
        "trajectory_format": "pt",
        "size": 1,
        "total_samples": int(num_samples),
        "trajectory_counter": 1,
        "seed": int(seed),
    }
    (output_rank_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    trajectory_index = {
        "trajectory_index": {
            str(trajectory_id): {
                "num_samples": int(num_samples),
                "trajectory_id": int(trajectory_id),
                "max_episode_length": int(payload["max_episode_length"]),
                "shape": [int(num_samples), 1, int(chunk_size)],
                "model_weights_id": model_weights_id,
            }
        },
        "trajectory_id_list": [int(trajectory_id)],
    }
    (output_rank_dir / "trajectory_index.json").write_text(
        json.dumps(trajectory_index, indent=2) + "\n"
    )


def main() -> None:
    args = _build_parser().parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_rank_dir = args.output_rank_dir.expanduser().resolve()

    states, main_images, wrist_images, extra_view_images = _load_frames(
        source_dir,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    payload = _build_trajectory_payload(
        source_dir=source_dir,
        states=states,
        main_images=main_images,
        wrist_images=wrist_images,
        extra_view_images=extra_view_images,
        chunk_size=args.chunk_size,
    )
    num_samples = int(payload["actions"].shape[0])
    _write_rank_dir(
        output_rank_dir=output_rank_dir,
        trajectory_id=args.trajectory_id,
        payload=payload,
        num_samples=num_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    trajectory_file = (
        output_rank_dir
        / f"trajectory_{args.trajectory_id}_{payload['model_weights_id']}.pt"
    )
    print(f"source_dir={source_dir}")
    print(f"output_rank_dir={output_rank_dir}")
    print(f"num_frames={states.shape[0]}")
    print(f"num_samples={num_samples}")
    print(f"trajectory_file={trajectory_file}")


if __name__ == "__main__":
    main()
