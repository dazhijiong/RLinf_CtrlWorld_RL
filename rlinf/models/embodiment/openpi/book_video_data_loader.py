from __future__ import annotations

import logging

import openpi.training.data_loader as openpi_data_loader
import openpi.transforms as _transforms
import torch
from openpi.training import config as _config

from rlinf.data.datasets.lerobot_book import (
    LeRobotBookDataset,
    LeRobotBookDatasetMetadata,
)


def create_book_data_loader(
    config: _config.TrainConfig,
    *,
    shuffle: bool = False,
    skip_norm_stats: bool = False,
    framework: str = "pytorch",
) -> openpi_data_loader.DataLoader:
    """OpenPI adapter for the local image-backed book dataset."""
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"book data_config: {data_config}")

    dataset_meta = LeRobotBookDatasetMetadata(data_config.repo_id)
    frame_stride = max(1, int(getattr(config.data, "frame_stride", 1)))
    camera_keys = getattr(config.data, "camera_keys", None)
    dataset = LeRobotBookDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [
                (t * frame_stride) / dataset_meta.fps
                for t in range(config.model.action_horizon)
            ]
            for key in data_config.action_sequence_keys
        },
        frame_stride=frame_stride,
        camera_keys=camera_keys,
    )

    if data_config.prompt_from_task:
        dataset = openpi_data_loader.TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    dataset = openpi_data_loader.transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats
    )

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = config.batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = config.batch_size

        torch_loader = openpi_data_loader.TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=config.num_workers,
            seed=config.seed,
            framework=framework,
        )
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    return openpi_data_loader.DataLoaderImpl(data_config, torch_loader)
