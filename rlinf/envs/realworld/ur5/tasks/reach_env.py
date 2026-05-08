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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ur5_env import UR5Env, UR5RobotConfig


@dataclass
class UR5ReachConfig(UR5RobotConfig):
    """Default reach-style task for UR5 realworld integration."""

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.0, 0.20, 3.14, 0.0, 0.0])
    )
    reset_ee_pose: Optional[np.ndarray] = None
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.02
    random_rz_range: float = np.pi / 12
    clip_x_range: float = 0.05
    clip_y_range: float = 0.05
    clip_z_range_low: float = 0.02
    clip_z_range_high: float = 0.08
    clip_rz_range: float = np.pi / 6
    enable_random_reset: bool = True
    enable_gripper_penalty: bool = False

    def __post_init__(self):
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reward_threshold = np.array(self.reward_threshold)
        if self.reset_ee_pose is None:
            self.reset_ee_pose = self.target_ee_pose.copy()
        else:
            self.reset_ee_pose = np.array(self.reset_ee_pose)
        self.action_scale = np.array([0.02, 0.1, 1.0])
        self.ee_pose_limit_min = np.array(
            [
                self.target_ee_pose[0] - self.clip_x_range,
                self.target_ee_pose[1] - self.clip_y_range,
                self.target_ee_pose[2] - self.clip_z_range_low,
                self.target_ee_pose[3] - 0.01,
                self.target_ee_pose[4] - 0.01,
                self.target_ee_pose[5] - self.clip_rz_range,
            ]
        )
        self.ee_pose_limit_max = np.array(
            [
                self.target_ee_pose[0] + self.clip_x_range,
                self.target_ee_pose[1] + self.clip_y_range,
                self.target_ee_pose[2] + self.clip_z_range_high,
                self.target_ee_pose[3] + 0.01,
                self.target_ee_pose[4] + 0.01,
                self.target_ee_pose[5] + self.clip_rz_range,
            ]
        )


class UR5ReachEnv(UR5Env):
    """Reference UR5 task that can be copied into task-specific envs."""

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = UR5ReachConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "Open the blue book"
