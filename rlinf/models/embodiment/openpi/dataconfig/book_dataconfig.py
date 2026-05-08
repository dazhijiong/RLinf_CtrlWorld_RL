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
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import ur5_realworld_policy


@dataclasses.dataclass(frozen=True)
class LeRobotBookDataConfig(DataConfigFactory):
    default_prompt: str | None = None
    extra_delta_transform: bool = True
    frame_stride: int = 1
    camera_keys: tuple[str, ...] = (
        "observation.images.d405_rgb",
        "observation.images.d435_rgb",
        "observation.images.d405_1_rgb",
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        camera_keys = tuple(self.camera_keys)
        if not 1 <= len(camera_keys) <= 3:
            raise ValueError(
                f"Expected 1 to 3 camera_keys for book data, got {camera_keys}"
            )

        repack_mapping = {
            "observation/image": camera_keys[0],
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if len(camera_keys) >= 2:
            repack_mapping["observation/wrist_image"] = camera_keys[1]
        if len(camera_keys) >= 3:
            repack_mapping["observation/extra_view_image"] = camera_keys[2]

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    repack_mapping
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                ur5_realworld_policy.UR5RealWorldInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[ur5_realworld_policy.UR5RealWorldOutputs()],
        )

        if not self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
