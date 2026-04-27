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

import socket
import time
from typing import Optional

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .ur5_robot_state import UR5RobotState


class RobotiqSocketClient:
    """Minimal socket client for the Robotiq gripper URCap service."""

    def __init__(self, host: str, port: int = 63352, timeout: float = 2.0):
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._socket.settimeout(timeout)

    def _send(self, command: str) -> str:
        self._socket.sendall((command.strip() + "\n").encode("ascii"))
        return self._socket.recv(1024).decode("ascii", errors="ignore").strip()

    def activate(self):
        self._send("SET ACT 1")
        self._send("SET GTO 1")
        time.sleep(0.5)

    def move(self, position: int, speed: int = 255, force: int = 150):
        position = int(np.clip(position, 0, 255))
        speed = int(np.clip(speed, 0, 255))
        force = int(np.clip(force, 0, 255))
        self._send(f"SET POS {position}")
        self._send(f"SET SPE {speed}")
        self._send(f"SET FOR {force}")
        self._send("SET GTO 1")

    def open(self):
        self.move(0)

    def close(self):
        self.move(255)

    def disconnect(self):
        self._socket.close()


class CustomUR5Controller(Worker):
    """UR5 controller adapter backed by the ``ur-rtde`` package.

    This adapter is intentionally conservative: it covers the minimum interface
    that ``UR5Env`` expects while keeping hardware-specific choices easy to edit
    in one place.
    """

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        gripper_type: str = "robotiq",
        gripper_port: int = 63352,
        tcp_offset: Optional[list[float]] = None,
        move_acc: float = 0.25,
        move_vel: float = 0.25,
    ):
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return CustomUR5Controller.create_group(
            robot_ip, gripper_type, gripper_port, tcp_offset, move_acc, move_vel
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"CustomUR5Controller-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        gripper_type: str = "robotiq",
        gripper_port: int = 63352,
        tcp_offset: Optional[list[float]] = None,
        move_acc: float = 0.25,
        move_vel: float = 0.25,
    ):
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._gripper_type = gripper_type.lower()
        self._gripper_port = gripper_port
        self._tcp_offset = tcp_offset
        self._move_acc = move_acc
        self._move_vel = move_vel
        self._state = UR5RobotState()
        self._gripper_open = False
        self._robotiq_client: Optional[RobotiqSocketClient] = None

        from dashboard_client import DashboardClient
        from rtde_control import RTDEControlInterface
        from rtde_io import RTDEIOInterface
        from rtde_receive import RTDEReceiveInterface

        self._rtde_control = RTDEControlInterface(robot_ip)
        self._rtde_receive = RTDEReceiveInterface(robot_ip)
        self._rtde_io: Optional[RTDEIOInterface] = None
        self._dashboard: Optional[DashboardClient] = None

        try:
            self._rtde_io = RTDEIOInterface(robot_ip)
        except Exception as exc:
            self._logger.warning(
                f"Failed to initialize RTDEIOInterface for UR5 at {robot_ip}: {exc}"
            )

        try:
            self._dashboard = DashboardClient(robot_ip)
            self._dashboard.connect()
        except Exception as exc:
            self._logger.warning(
                f"Failed to initialize DashboardClient for UR5 at {robot_ip}: {exc}"
            )

        if self._tcp_offset is not None:
            try:
                self._rtde_control.setTcp(self._tcp_offset)
            except Exception as exc:
                self._logger.warning(
                    f"Failed to set TCP offset {self._tcp_offset} for UR5 at {robot_ip}: {exc}"
                )

        if self._gripper_type == "robotiq":
            try:
                self._robotiq_client = RobotiqSocketClient(
                    robot_ip, port=self._gripper_port
                )
                self._robotiq_client.activate()
            except Exception as exc:
                self._logger.warning(
                    f"Failed to initialize Robotiq gripper at {robot_ip}:{self._gripper_port}: {exc}"
                )

    def is_robot_up(self) -> bool:
        return self._rtde_control.isConnected() and self._rtde_receive.isConnected()

    def clear_errors(self):
        if self._dashboard is None:
            return False

        try:
            self._dashboard.closeSafetyPopup()
        except Exception:
            pass
        try:
            self._dashboard.unlockProtectiveStop()
        except Exception:
            pass
        return True

    def reconfigure_compliance_params(self, params: dict[str, float]):
        # ur_rtde does not expose Franka-style compliance tuning. Keep this as a
        # no-op so the realworld env can share the same control flow.
        return params

    def reset_joint(self, reset_pos: list[float]):
        self._rtde_control.moveJ(reset_pos, self._move_vel, self._move_acc)
        return True

    def move_arm(self, pose: np.ndarray):
        pose = np.asarray(pose, dtype=np.float64)
        self._rtde_control.moveL(pose, self._move_vel, self._move_acc)
        return True

    def open_gripper(self):
        self._gripper_open = True
        if self._gripper_type == "robotiq" and self._robotiq_client is not None:
            try:
                self._robotiq_client.open()
            except Exception as exc:
                self._logger.warning(f"Failed to open Robotiq gripper: {exc}")
        elif self._rtde_io is not None:
            try:
                self._rtde_io.setToolDigitalOut(0, True)
            except Exception as exc:
                self._logger.warning(f"Failed to set UR5 gripper open IO: {exc}")
        self._state.gripper_position = 1.0
        self._state.gripper_open = True
        return True

    def close_gripper(self):
        self._gripper_open = False
        if self._gripper_type == "robotiq" and self._robotiq_client is not None:
            try:
                self._robotiq_client.close()
            except Exception as exc:
                self._logger.warning(f"Failed to close Robotiq gripper: {exc}")
        elif self._rtde_io is not None:
            try:
                self._rtde_io.setToolDigitalOut(0, False)
            except Exception as exc:
                self._logger.warning(f"Failed to set UR5 gripper close IO: {exc}")
        self._state.gripper_position = 0.0
        self._state.gripper_open = False
        return True

    def get_state(self) -> UR5RobotState:
        pose = np.asarray(self._rtde_receive.getActualTCPPose(), dtype=np.float64)
        self._state.tcp_pose = pose.astype(np.float32)

        self._state.tcp_vel = np.asarray(
            self._rtde_receive.getActualTCPSpeed(), dtype=np.float32
        )
        self._state.arm_joint_position = np.asarray(
            self._rtde_receive.getActualQ(), dtype=np.float32
        )
        self._state.arm_joint_velocity = np.asarray(
            self._rtde_receive.getActualQd(), dtype=np.float32
        )

        tcp_wrench = np.asarray(
            self._rtde_receive.getActualTCPForce(), dtype=np.float32
        )
        if tcp_wrench.shape[0] >= 6:
            self._state.tcp_force = tcp_wrench[:3]
            self._state.tcp_torque = tcp_wrench[3:6]

        self._state.gripper_open = self._gripper_open
        self._state.gripper_position = 1.0 if self._gripper_open else 0.0
        return self._state

    def __del__(self):
        try:
            if self._dashboard is not None:
                self._dashboard.disconnect()
        except Exception:
            pass
        try:
            self._rtde_control.disconnect()
        except Exception:
            pass
        try:
            self._rtde_receive.disconnect()
        except Exception:
            pass
        try:
            if self._rtde_io is not None:
                self._rtde_io.disconnect()
        except Exception:
            pass
        try:
            if self._robotiq_client is not None:
                self._robotiq_client.disconnect()
        except Exception:
            pass
