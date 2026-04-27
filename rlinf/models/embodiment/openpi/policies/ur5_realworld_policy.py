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

import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _extract_ur5_ee_state(state) -> np.ndarray:
    state = np.asarray(state)
    if state.shape == (7,):
        return state
    if state.shape == (19,):
        gripper = state[:1]
        tcp_pose = state[4:10]
        return np.concatenate((tcp_pose, gripper), axis=0)
    raise ValueError(
        f"Expected UR5 state shape (7,) or (19,), but got {state.shape}."
    )


def _extract_extra_views(
    extra_view_image, base_image: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.bool_, np.bool_]]:
    if extra_view_image is None:
        empty = np.zeros_like(base_image)
        return (empty, empty), (np.False_, np.False_)

    extra_view_image = np.asarray(extra_view_image)
    if extra_view_image.ndim == 3:
        extra_view_image = extra_view_image[None, ...]
    if extra_view_image.ndim != 4:
        raise ValueError(
            "Expected extra view image to have shape (N, H, W, C) or (H, W, C), "
            f"but got {extra_view_image.shape}."
        )

    parsed_images = []
    image_masks = []
    for idx in range(2):
        if idx < extra_view_image.shape[0]:
            parsed_images.append(_parse_image(extra_view_image[idx]))
            image_masks.append(np.True_)
        else:
            parsed_images.append(np.zeros_like(base_image))
            image_masks.append(np.False_)
    return tuple(parsed_images), tuple(image_masks)


def _extract_named_view(image, base_image: np.ndarray) -> tuple[np.ndarray, np.bool_]:
    if image is None:
        return np.zeros_like(base_image), np.False_
    return _parse_image(image), np.True_


@dataclasses.dataclass(frozen=True)
class UR5RealWorldOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class UR5RealWorldInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = _extract_ur5_ee_state(data["observation/state"])
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/image"])
        extra_view_image = data.get("observation/extra_view_image")
        use_named_views = "observation/wrist_image" in data
        if not use_named_views and extra_view_image is not None:
            extra_view_ndim = np.asarray(extra_view_image).ndim
            use_named_views = extra_view_ndim == 3

        if use_named_views:
            left_wrist, left_mask = _extract_named_view(
                data.get("observation/wrist_image"), base_image
            )
            right_wrist, right_mask = _extract_named_view(
                extra_view_image, base_image
            )
            extra_images = (left_wrist, right_wrist)
            extra_masks = (left_mask, right_mask)
        else:
            extra_images, extra_masks = _extract_extra_views(
                extra_view_image, base_image
            )

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, *extra_images)
            image_masks = (np.True_, *extra_masks)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, *extra_images)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            assert len(data["actions"].shape) == 2 and data["actions"].shape[-1] == 7, (
                f"Expected actions shape (N, 7), got {data['actions'].shape}"
            )
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
