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

import copy
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.envs.realworld.franka.utils import (
    clip_euler_to_target_window,
    quat_slerp,
)
from rlinf.scheduler import UR5HWInfo, WorkerInfo
from rlinf.utils.logging import get_logger

from .ur5_controller import UR5Controller
from .ur5_robot_state import UR5RobotState


@dataclass
class UR5RobotConfig:
    robot_ip: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    enable_camera_player: bool = True

    # For real deployment, point this to a custom adapter:
    # "your_package.your_module:YourUR5Controller"
    controller_cls_path: Optional[str] = None
    controller_backend: str = "mock"
    gripper_type: str = "robotiq"
    gripper_port: int = 63352
    tcp_offset: Optional[list[float]] = None
    move_acc: float = 0.25
    move_vel: float = 0.25

    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.0, 0.20, 3.14, 0.0, 0.0])
    )
    reset_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.0, 0.28, 3.14, 0.0, 0.0])
    )
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    action_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    enable_random_reset: bool = False
    random_xy_range: float = 0.0
    random_rz_range: float = 0.0

    ee_pose_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ee_pose_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(6))
    compliance_param: dict[str, float] = field(default_factory=dict)
    precision_param: dict[str, float] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None
    success_hold_steps: int = 1


class UR5Env(gym.Env):
    """UR5 robot arm environment following RLinf's realworld contract."""

    def __init__(
        self,
        config: UR5RobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[UR5HWInfo],
        env_idx: int,
    ):
        self._logger = get_logger()
        self.config = config
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._ur5_state = UR5RobotState()
        if not self.config.is_dummy:
            self._reset_pose = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:].copy()).as_quat(),
                ]
            ).copy()
        else:
            self._reset_pose = np.zeros(7)

        self._num_steps = 0
        self._success_hold_counter = 0

        if not self.config.is_dummy:
            self._setup_hardware()

        if self.config.camera_serials is None:
            self.config.camera_serials = []

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        start_time = time.time()
        while not self._controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time} seconds for UR5 robot to be ready."
                )

        self._interpolate_move(self._reset_pose)
        time.sleep(0.5)
        self._ur5_state = self._controller.get_state().wait()[0]
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _setup_hardware(self):
        assert self.env_idx >= 0, "env_idx must be set for UR5Env."
        assert isinstance(self.hardware_info, UR5HWInfo), (
            f"hardware_info must be UR5HWInfo, but got {type(self.hardware_info)}."
        )

        if self.config.robot_ip is None:
            self.config.robot_ip = self.hardware_info.config.robot_ip
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials

        assert self.config.controller_cls_path is not None, (
            "UR5 real deployment requires 'controller_cls_path' in override_cfg. "
            "Point it to a custom controller adapter, for example "
            "'your_package.your_module:YourUR5Controller'."
        )

        self._controller = UR5Controller.launch_controller(
            robot_ip=self.config.robot_ip,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
            controller_backend=self.config.controller_backend,
            controller_cls_path=self.config.controller_cls_path,
            gripper_type=self.config.gripper_type,
            gripper_port=self.config.gripper_port,
            tcp_offset=self.config.tcp_offset,
            move_acc=self.config.move_acc,
            move_vel=self.config.move_vel,
        )

    def step(self, action: np.ndarray):
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.next_position = self._ur5_state.tcp_pose.copy()
        self.next_position[:3] = (
            self.next_position[:3] + xyz_delta * self.config.action_scale[0]
        )

        if not self.config.is_dummy:
            self.next_position[3:] = (
                R.from_euler("xyz", action[3:6] * self.config.action_scale[1])
                * R.from_quat(self._ur5_state.tcp_pose[3:].copy())
            ).as_quat()

            gripper_action = action[6] * self.config.action_scale[2]
            is_gripper_action_effective = self._gripper_action(gripper_action)
            self._move_action(self._clip_position_to_safety_box(self.next_position))
        else:
            is_gripper_action_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._ur5_state = self._controller.get_state().wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_action_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray | UR5RobotState],
        is_gripper_action_effective: bool = False,
    ) -> float:
        if self.config.is_dummy:
            return 0.0

        euler_angles = np.abs(
            R.from_quat(self._ur5_state.tcp_pose[3:].copy()).as_euler("xyz")
        )
        position = np.hstack([self._ur5_state.tcp_pose[:3], euler_angles])
        target_delta = np.abs(position - self.config.target_ee_pose)
        is_in_target_zone = np.all(target_delta[:3] <= self.config.reward_threshold[:3])

        if is_in_target_zone:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = np.exp(-500 * np.sum(np.square(target_delta[:3])))
            else:
                reward = 0.0

        if self.config.enable_gripper_penalty and is_gripper_action_effective:
            reward -= self.config.gripper_penalty
        return reward

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            return self._get_observation(), {}

        self._success_hold_counter = 0
        self._controller.reconfigure_compliance_params(
            self.config.compliance_param
        ).wait()
        self.go_to_rest(joint_reset)
        self._clear_error()
        self._num_steps = 0
        self._ur5_state = self._controller.get_state().wait()[0]
        return self._get_observation(), {}

    def go_to_rest(self, joint_reset=False):
        if joint_reset:
            self._controller.reset_joint(self.config.joint_reset_qpos).wait()
            time.sleep(0.5)

        if self.config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose[:2] += np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            euler_random = self.config.target_ee_pose[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range
            )
            reset_pose[3:] = R.from_euler("xyz", euler_random).as_quat()
        else:
            reset_pose = self._reset_pose.copy()

        self._ur5_state = self._controller.get_state().wait()[0]
        self._interpolate_move(reset_pose)

    def _init_action_obs_spaces(self):
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[:3],
            high=self.config.ee_pose_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[3:],
            high=self.config.ee_pose_limit_max[3:],
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(len(self.config.camera_serials))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _open_cameras(self):
        self._cameras: list[Camera] = []
        camera_infos = [
            CameraInfo(name=f"wrist_{i + 1}", serial_number=n)
            for i, n in enumerate(self.config.camera_serials)
        ]
        for info in camera_infos:
            camera = Camera(info)
            camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> np.ndarray:
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]
        resized_frame = cv2.resize(cropped_frame, reshape_size)
        return cropped_frame, resized_frame

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        frames = {}
        display_frames = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                reshape_size = self.observation_space["frames"][
                    camera._camera_info.name
                ].shape[:2][::-1]
                cropped_frame, resized_frame = self._crop_frame(frame, reshape_size)
                frames[camera._camera_info.name] = resized_frame[..., ::-1]
                display_frames[camera._camera_info.name] = resized_frame
                display_frames[f"{camera._camera_info.name}_full"] = cropped_frame
            except queue.Empty:
                self._logger.warning(
                    f"Camera {camera._camera_info.name} is not producing frames."
                )
                raise

        self.camera_player.put_frame(display_frames)
        return frames

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        position[:3] = np.clip(
            position[:3], self._xyz_safe_space.low, self._xyz_safe_space.high
        )
        euler = R.from_quat(position[3:].copy()).as_euler("xyz")
        euler = clip_euler_to_target_window(
            euler=euler,
            target_euler=self.config.target_ee_pose[3:],
            lower_euler=self._rpy_safe_space.low,
            upper_euler=self._rpy_safe_space.high,
        )
        position[3:] = R.from_euler("xyz", euler).as_quat()
        return position

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _gripper_action(self, position: float, is_binary: bool = True):
        if not is_binary:
            raise NotImplementedError("Non-binary gripper action is not implemented.")
        if (
            position <= -self.config.binary_gripper_threshold
            and self._ur5_state.gripper_open
        ):
            self._controller.close_gripper().wait()
            time.sleep(0.2)
            return True
        if (
            position >= self.config.binary_gripper_threshold
            and not self._ur5_state.gripper_open
        ):
            self._controller.open_gripper().wait()
            time.sleep(0.2)
            return True
        return False

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.0):
        num_steps = int(timeout * self.config.step_frequency)
        self._ur5_state = self._controller.get_state().wait()[0]
        pos_path = np.linspace(self._ur5_state.tcp_pose[:3], pose[:3], num_steps + 1)
        quat_path = quat_slerp(self._ur5_state.tcp_pose[3:], pose[3:], num_steps + 1)
        for pos, quat in zip(pos_path[1:], quat_path[1:]):
            interpolated_pose = np.concatenate([pos, quat])
            self._move_action(interpolated_pose.astype(np.float32))
            time.sleep(1.0 / self.config.step_frequency)
        self._ur5_state = self._controller.get_state().wait()[0]

    def _move_action(self, position: np.ndarray):
        self._clear_error()
        self._controller.move_arm(position.astype(np.float32)).wait()

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        frames = self._get_camera_frames()
        state = {
            "tcp_pose": self._ur5_state.tcp_pose,
            "tcp_vel": self._ur5_state.tcp_vel,
            "gripper_position": np.array([self._ur5_state.gripper_position]),
            "tcp_force": self._ur5_state.tcp_force,
            "tcp_torque": self._ur5_state.tcp_torque,
        }
        return copy.deepcopy({"state": state, "frames": frames})
