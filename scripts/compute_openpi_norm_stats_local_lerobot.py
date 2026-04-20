#!/usr/bin/env python3
"""Compute OpenPI-compatible norm stats for a local LeRobot dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi import transforms as openpi_transforms
from openpi.shared import normalize as openpi_normalize
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def _compute_array_stats(values: np.ndarray, target_dim: int) -> openpi_normalize.NormStats:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    values = values.reshape(-1, values.shape[-1])

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    q01 = np.quantile(values, 0.01, axis=0)
    q99 = np.quantile(values, 0.99, axis=0)

    def _pad(vec: np.ndarray) -> np.ndarray:
        if vec.shape[-1] > target_dim:
            raise ValueError(f"Vector dim {vec.shape[-1]} exceeds target dim {target_dim}.")
        if vec.shape[-1] == target_dim:
            return vec.astype(np.float32)
        return np.pad(vec, (0, target_dim - vec.shape[-1]), mode="constant").astype(np.float32)

    return openpi_normalize.NormStats(
        mean=_pad(mean),
        std=_pad(std),
        q01=_pad(q01),
        q99=_pad(q99),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--repo-id", default="physical-intelligence/libero")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / args.repo_id
    )

    os.environ["HF_LEROBOT_HOME"] = str(dataset_root)

    train_cfg = get_openpi_config(args.config_name, model_path=str(model_path))
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    dataset_meta = LeRobotDatasetMetadata(args.repo_id)
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(train_cfg.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    transform_fns = []
    if data_config.prompt_from_task:
        transform_fns.append(openpi_transforms.PromptFromLeRobotTask(dataset_meta.tasks))
    transform_fns.extend(data_config.repack_transforms.inputs)
    transform_fns.extend(data_config.data_transforms.inputs)

    states = []
    actions = []
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        for transform_fn in transform_fns:
            sample = transform_fn(sample)
        states.append(np.asarray(sample["state"], dtype=np.float32))
        actions.append(np.asarray(sample["actions"], dtype=np.float32))

    states_arr = np.stack(states, axis=0)
    actions_arr = np.stack(actions, axis=0)
    target_dim = train_cfg.model.action_dim

    norm_stats = {
        "state": _compute_array_stats(states_arr, target_dim),
        "actions": _compute_array_stats(actions_arr, target_dim),
    }
    openpi_normalize.save(output_dir, norm_stats)

    print(f"Saved norm stats to: {output_dir / 'norm_stats.json'}")
    print(f"Dataset root: {dataset_root}")
    print(f"Repo id: {args.repo_id}")
    print(f"Num samples: {len(dataset)}")
    print(f"State shape before pad: {states_arr.shape}")
    print(f"Action shape before pad: {actions_arr.shape}")


if __name__ == "__main__":
    main()
