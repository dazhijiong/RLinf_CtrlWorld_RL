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

import importlib
from typing import Any, Optional

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker

from .ur5_robot_state import UR5RobotState


def _import_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path '{import_path}'. Expected format 'package.module:Object'."
        )
    module_name, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


class UR5Controller(Worker):
    """Minimal UR5 controller scaffold.

    The built-in controller intentionally only supports a mock backend so RLinf can
    register the robot and run dummy smoke tests. Real deployment should provide a
    custom adapter via ``controller_cls_path`` that implements a compatible
    ``launch_controller(...)`` classmethod or staticmethod.
    """

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        controller_backend: str = "mock",
        controller_cls_path: Optional[str] = None,
        gripper_type: str = "robotiq",
        gripper_port: int = 63352,
        tcp_offset: Optional[list[float]] = None,
        move_acc: float = 0.25,
        move_vel: float = 0.25,
    ):
        if controller_cls_path:
            controller_cls = _import_object(controller_cls_path)
            if not hasattr(controller_cls, "launch_controller"):
                raise AttributeError(
                    f"Custom UR5 controller '{controller_cls_path}' must provide launch_controller(...)."
                )
            return controller_cls.launch_controller(
                robot_ip=robot_ip,
                env_idx=env_idx,
                node_rank=node_rank,
                worker_rank=worker_rank,
                gripper_type=gripper_type,
                gripper_port=gripper_port,
                tcp_offset=tcp_offset,
                move_acc=move_acc,
                move_vel=move_vel,
            )

        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return UR5Controller.create_group(robot_ip, controller_backend).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"UR5Controller-{worker_rank}-{env_idx}",
        )

    def __init__(self, robot_ip: str, controller_backend: str = "mock"):
        super().__init__()
        self._robot_ip = robot_ip
        self._controller_backend = controller_backend
        self._state = UR5RobotState()

        if self._controller_backend != "mock":
            raise NotImplementedError(
                "Built-in UR5Controller only supports the 'mock' backend. "
                "Provide controller_cls_path in the env override config to use a real UR5 adapter."
            )

    def is_robot_up(self) -> bool:
        return True

    def get_state(self) -> UR5RobotState:
        return self._state

    def clear_errors(self):
        return True

    def reconfigure_compliance_params(self, params: dict[str, float]):
        return params

    def reset_joint(self, reset_pos: list[float]):
        self._state.arm_joint_position = np.array(reset_pos, dtype=np.float32)
        return True

    def move_arm(self, pose: np.ndarray):
        pose = np.array(pose, dtype=np.float32)
        self._state.tcp_pose = pose.copy()
        return True

    def open_gripper(self):
        self._state.gripper_open = True
        self._state.gripper_position = 1.0
        return True

    def close_gripper(self):
        self._state.gripper_open = False
        self._state.gripper_position = 0.0
        return True
