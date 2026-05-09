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
from scipy.spatial.transform import Rotation as R, Slerp

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
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
    control_mode: str = "joint"
    use_servo: bool = True
    servo_time: Optional[float] = None
    servo_lookahead_time: float = 0.1
    servo_gain: int = 600
    reset_servo_duration: float = 3.0
    reset_servo_vel: float = 0.03
    reset_servo_acc: float = 0.05

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
        self.config.control_mode = self.config.control_mode.lower()
        if self.config.control_mode not in {"joint", "pose"}:
            raise ValueError(
                f"Unsupported UR5 control_mode: {self.config.control_mode}. "
                "Expected 'joint' or 'pose'."
            )
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._ur5_state = UR5RobotState()
        if not self.config.is_dummy:
            self._reset_pose = self._to_rtde_pose(self.config.reset_ee_pose).copy()
        else:
            self._reset_pose = np.zeros(6)

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

        self._servo_reset_to_pose(self._reset_pose)
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
            use_servo=self.config.use_servo,
            servo_time=self.config.servo_time
            if self.config.servo_time is not None
            else 1.0 / self.config.step_frequency,
            servo_lookahead_time=self.config.servo_lookahead_time,
            servo_gain=self.config.servo_gain,
        )

    def _to_rtde_pose(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,):
            raise ValueError(
                f"UR5 poses must be shape (6,) as [x, y, z, rx, ry, rz], got {pose.shape}."
            )
        return pose.copy()

    def step(self, action: np.ndarray):
        start_time = time.time()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        target = action[:6].copy()

        if not self.config.is_dummy:
            self._ur5_state = self._controller.get_state().wait()[0]
            gripper_action = float(action[6])
            is_gripper_action_effective = self._gripper_action(gripper_action)
            if self.config.control_mode == "joint":
                self._move_joint_action(target)
            else:
                self._move_pose_action(target)
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

        target_pose = self._to_rtde_pose(self.config.target_ee_pose)
        position = self._ur5_state.tcp_pose.copy()
        target_delta = np.abs(position - target_pose)
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

    def get_current_obs(self):
        if not self.config.is_dummy:
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
            if self.config.random_rz_range != 0:
                self._logger.warning(
                    "Ignoring random_rz_range for rotvec reset pose orientation."
                )
        else:
            reset_pose = self._reset_pose.copy()

        self._servo_reset_to_pose(reset_pose)

    def _servo_reset_to_pose(self, pose: np.ndarray):
        self._ur5_state = self._controller.get_state().wait()[0]
        start_pose = self._ur5_state.tcp_pose.copy()
        self._logger.warning(
            "Starting UR5 servo reset: current_pose=%s target_pose=%s "
            "duration=%.3f velocity=%.3f acceleration=%.3f",
            np.round(start_pose, 6).tolist(),
            np.round(pose, 6).tolist(),
            self.config.reset_servo_duration,
            self.config.reset_servo_vel,
            self.config.reset_servo_acc,
        )
        num_steps = max(
            2, int(round(self.config.reset_servo_duration * self.config.step_frequency))
        )
        pos_path = np.linspace(start_pose[:3], pose[:3], num_steps + 1)
        key_rots = R.from_rotvec(np.stack([start_pose[3:], pose[3:]], axis=0))
        slerp = Slerp([0.0, 1.0], key_rots)
        rotvec_path = slerp(np.linspace(0.0, 1.0, num_steps + 1)).as_rotvec()
        step_duration = self.config.reset_servo_duration / num_steps
        for pos, rotvec in zip(pos_path[1:], rotvec_path[1:]):
            waypoint = np.concatenate([pos, rotvec]).astype(np.float32)
            self._controller.servo_arm(
                waypoint,
                duration=step_duration,
                velocity=self.config.reset_servo_vel,
                acceleration=self.config.reset_servo_acc,
            ).wait()
        self._ur5_state = self._controller.get_state().wait()[0]
        pose_error = np.asarray(self._ur5_state.tcp_pose) - np.asarray(pose)
        self._logger.warning(
            "Finished UR5 servo reset: current_pose=%s target_pose=%s error=%s",
            np.round(self._ur5_state.tcp_pose, 6).tolist(),
            np.round(pose, 6).tolist(),
            np.round(pose_error, 6).tolist(),
        )

    def _init_action_obs_spaces(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-np.inf] * 6 + [0.0], dtype=np.float32),
            high=np.array([np.inf] * 6 + [1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),
                        "joint_velocity": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),
                        "gripper_position": gym.spaces.Box(0, 1, shape=(1,)),
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

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _gripper_action(self, position: float, is_binary: bool = True):
        if not is_binary:
            raise NotImplementedError("Non-binary gripper action is not implemented.")
        if (
            position <= self.config.binary_gripper_threshold
            and not self._ur5_state.gripper_open
        ):
            self._controller.open_gripper().wait()
            time.sleep(0.2)
            return True
        if (
            position > self.config.binary_gripper_threshold
            and self._ur5_state.gripper_open
        ):
            self._controller.close_gripper().wait()
            time.sleep(0.2)
            return True
        return False

    def _move_joint_action(self, joint_position: np.ndarray):
        self._clear_error()
        self._controller.reset_joint(joint_position.astype(np.float32)).wait()

    def _move_pose_action(self, pose: np.ndarray):
        self._clear_error()
        self._controller.move_arm(self._to_rtde_pose(pose).astype(np.float32)).wait()

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        frames = self._get_camera_frames()
        state = {
            "tcp_pose": self._ur5_state.tcp_pose,
            "tcp_vel": self._ur5_state.tcp_vel,
            "joint_position": self._ur5_state.arm_joint_position,
            "joint_velocity": self._ur5_state.arm_joint_velocity,
            "gripper_position": np.array([self._ur5_state.gripper_position]),
            "tcp_force": self._ur5_state.tcp_force,
            "tcp_torque": self._ur5_state.tcp_torque,
        }
        return copy.deepcopy({"state": state, "frames": frames})
