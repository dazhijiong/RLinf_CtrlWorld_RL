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

"""Replay UR5 end-effector trajectories from an RLinf replay buffer.

This bypasses pi05 entirely. It reads flattened action chunks from replay-buffer
trajectory files and replays absolute ``[x, y, z, r1, r2, r3, gripper]``
waypoints. Orientation can be interpreted either as Euler angles or as the UR
axis-angle / rotvec convention used by ``getActualTCPPose()``.

The script defaults to dry-run. Add ``--execute`` to move the robot.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.ur5.ur_rtde_controller import CustomUR5Controller


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _resolve_trajectory_path(
    replay_dir: Path,
    rank: str,
    trajectory_id: int,
) -> Path:
    rank_dir = replay_dir / rank
    if not rank_dir.is_dir():
        raise FileNotFoundError(f"Rank directory not found: {rank_dir}")

    matches = sorted(rank_dir.glob(f"trajectory_{trajectory_id}_*.pt"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find trajectory_{trajectory_id}_*.pt under {rank_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple matches for trajectory_id={trajectory_id} under {rank_dir}: {matches}"
        )
    return matches[0]


def _list_replay_buffer(replay_dir: Path) -> None:
    rank_dirs = sorted(p for p in replay_dir.iterdir() if p.is_dir() and p.name.startswith("rank_"))
    if not rank_dirs:
        raise FileNotFoundError(f"No rank_* directories found under {replay_dir}")

    print(f"Replay buffer: {replay_dir}")
    total = 0
    for rank_dir in rank_dirs:
        meta_path = rank_dir / "metadata.json"
        index_path = rank_dir / "trajectory_index.json"
        meta = _load_json(meta_path) if meta_path.exists() else {}
        index = _load_json(index_path) if index_path.exists() else {}
        traj_index = index.get("trajectory_index", {})
        ids = sorted(int(k) for k in traj_index.keys())
        total += len(ids)
        print(
            f"  {rank_dir.name}: size={meta.get('size', len(ids))}, "
            f"total_samples={meta.get('total_samples', 'unknown')}, ids={ids[:10]}"
            f"{'...' if len(ids) > 10 else ''}"
        )
    print(f"Total trajectories: {total}")


def _select_action_tensor(trajectory: dict, batch_index: int) -> np.ndarray:
    actions = trajectory["actions"]
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    actions = np.asarray(actions)
    if actions.ndim != 3:
        raise ValueError(f"Expected actions shape [T, B, flat], got {actions.shape}")
    if batch_index < 0 or batch_index >= actions.shape[1]:
        raise IndexError(
            f"batch_index {batch_index} out of range for action batch size {actions.shape[1]}"
        )
    return actions[:, batch_index]


def _reshape_chunks(flat_actions: np.ndarray, action_dim: int) -> np.ndarray:
    if flat_actions.shape[-1] % action_dim != 0:
        raise ValueError(
            f"Flattened action dim {flat_actions.shape[-1]} is not divisible by action_dim {action_dim}"
        )
    num_action_chunks = flat_actions.shape[-1] // action_dim
    return flat_actions.reshape(flat_actions.shape[0], num_action_chunks, action_dim)


def _load_waypoints(
    trajectory_path: Path,
    batch_index: int,
    action_dim: int,
) -> np.ndarray:
    trajectory = torch.load(trajectory_path, map_location="cpu")
    flat_actions = _select_action_tensor(trajectory, batch_index=batch_index)
    chunk_actions = _reshape_chunks(flat_actions, action_dim=action_dim)
    return chunk_actions.reshape(-1, action_dim)


def _load_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_pkl_waypoints(
    pkl_dir: Path,
    pkl_key: str,
    action_dim: int,
) -> np.ndarray:
    frame_paths = sorted(pkl_dir.glob("*.pkl"))
    if not frame_paths:
        raise FileNotFoundError(f"No .pkl files found under {pkl_dir}")

    waypoints = []
    for frame_path in frame_paths:
        frame = _load_pkl(frame_path)
        if pkl_key not in frame:
            raise KeyError(f"Missing key '{pkl_key}' in {frame_path}")
        waypoint = np.asarray(frame[pkl_key], dtype=np.float32)
        if waypoint.shape != (action_dim,):
            raise ValueError(
                f"Expected {pkl_key} shape {(action_dim,)}, got {waypoint.shape} in {frame_path}"
            )
        waypoints.append(waypoint)
    return np.stack(waypoints, axis=0)


def _summarize_waypoints(waypoints: np.ndarray) -> None:
    xyz = waypoints[:, :3]
    euler = waypoints[:, 3:6]
    gripper = waypoints[:, 6]
    print(f"Waypoints: {len(waypoints)}")
    print(f"  first: {np.round(waypoints[0], 6).tolist()}")
    print(f"  last : {np.round(waypoints[-1], 6).tolist()}")
    print(f"  xyz min: {np.round(xyz.min(axis=0), 6).tolist()}")
    print(f"  xyz max: {np.round(xyz.max(axis=0), 6).tolist()}")
    print(f"  euler min: {np.round(euler.min(axis=0), 6).tolist()}")
    print(f"  euler max: {np.round(euler.max(axis=0), 6).tolist()}")
    print(f"  gripper min/max: {gripper.min():.6f} / {gripper.max():.6f}")


def _pose_to_rotvec(
    pose_orientation: np.ndarray,
    orientation_mode: str,
) -> np.ndarray:
    if orientation_mode == "euler":
        return R.from_euler("xyz", pose_orientation).as_rotvec()
    if orientation_mode == "rotvec":
        return np.asarray(pose_orientation, dtype=np.float64)
    raise ValueError(f"Unsupported orientation_mode: {orientation_mode}")


def _interpolate_pose_sequence(
    start_pose_rotvec: np.ndarray,
    target_pose_rotvec: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    if num_steps <= 1:
        return [target_pose_rotvec.astype(np.float32)]
    pos_path = np.linspace(start_pose_rotvec[:3], target_pose_rotvec[:3], num_steps)
    key_rots = R.from_rotvec(np.stack([start_pose_rotvec[3:], target_pose_rotvec[3:]], axis=0))
    interp_rots = key_rots
    if num_steps > 2:
        from scipy.spatial.transform import Slerp

        slerp = Slerp([0.0, 1.0], key_rots)
        interp_rots = slerp(np.linspace(0.0, 1.0, num_steps))
    else:
        interp_rots = R.from_rotvec(np.stack([start_pose_rotvec[3:], target_pose_rotvec[3:]], axis=0))
    rotvecs = interp_rots.as_rotvec()
    if rotvecs.shape[0] != num_steps:
        rotvecs = np.vstack([start_pose_rotvec[3:], target_pose_rotvec[3:]])
    return [
        np.concatenate([pos, rotvec]).astype(np.float32)
        for pos, rotvec in zip(pos_path, rotvecs, strict=True)
    ]


def _maybe_apply_gripper(
    controller,
    gripper_value: float,
    threshold: float,
) -> None:
    # GELLO records gripper as 0=open, 1=closed.
    if gripper_value <= threshold:
        controller.open_gripper().wait()
    else:
        controller.close_gripper().wait()


def _execute_waypoints(
    controller,
    waypoints: np.ndarray,
    *,
    orientation_mode: str,
    gripper_threshold: float,
    initial_move_seconds: float,
    step_sleep: float,
) -> None:
    controller.clear_errors().wait()
    current_pose = np.asarray(controller.get_state().wait()[0].tcp_pose, dtype=np.float64)

    first_pose = waypoints[0, :6]
    first_pose_rotvec = np.concatenate(
        [
            first_pose[:3],
            _pose_to_rotvec(first_pose[3:6], orientation_mode=orientation_mode),
        ]
    )
    if initial_move_seconds > 0:
        num_steps = max(2, int(round(initial_move_seconds / max(step_sleep, 1e-3))))
        interp_path = _interpolate_pose_sequence(
            start_pose_rotvec=current_pose,
            target_pose_rotvec=first_pose_rotvec,
            num_steps=num_steps,
        )
        for pose in interp_path:
            controller.move_arm(pose).wait()
            time.sleep(step_sleep)
    else:
        controller.move_arm(first_pose_rotvec.astype(np.float32)).wait()
        time.sleep(step_sleep)

    for idx, waypoint in enumerate(waypoints):
        pose = waypoint[:6]
        gripper_value = float(waypoint[6])
        pose_rotvec = np.concatenate(
            [pose[:3], _pose_to_rotvec(pose[3:6], orientation_mode=orientation_mode)]
        ).astype(np.float32)
        controller.clear_errors().wait()
        controller.move_arm(pose_rotvec).wait()
        _maybe_apply_gripper(
            controller,
            gripper_value=gripper_value,
            threshold=gripper_threshold,
        )
        print(
            f"[{idx + 1:04d}/{len(waypoints):04d}] "
            f"xyz={np.round(pose[:3], 6).tolist()} "
            f"{orientation_mode}={np.round(pose[3:6], 6).tolist()} "
            f"gripper={gripper_value:.6f}"
        )
        time.sleep(step_sleep)


def _reset_robot_joints(
    controller,
    reset_joints: np.ndarray,
    settle_seconds: float,
) -> None:
    controller.clear_errors().wait()
    controller.reset_joint(reset_joints.astype(np.float32).tolist()).wait()
    if settle_seconds > 0:
        time.sleep(settle_seconds)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay UR5 absolute end-effector trajectories from an RLinf replay buffer."
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("/home/manager/Desktop/dagger_components/replay_buffer"),
        help="Replay buffer root containing rank_* subdirectories.",
    )
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        default=None,
        help="Direct path to a trajectory_*.pt file. Overrides --replay-dir/--rank/--trajectory-id.",
    )
    parser.add_argument(
        "--pkl-dir",
        type=Path,
        default=None,
        help="Directory of raw GELLO .pkl frames. Replays one frame at a time in filename order.",
    )
    parser.add_argument(
        "--pkl-key",
        default="ee_pos_quat",
        help="Frame key to replay from raw .pkl files. Default uses absolute EEF pose.",
    )
    parser.add_argument("--rank", default="rank_0", help="Rank directory under replay buffer.")
    parser.add_argument("--trajectory-id", type=int, default=0, help="Trajectory id inside the rank directory.")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index inside the stored trajectory.")
    parser.add_argument("--action-dim", type=int, default=7, help="Per-waypoint action dimension.")
    parser.add_argument(
        "--orientation-mode",
        choices=("euler", "rotvec"),
        default="rotvec",
        help="Interpret waypoint orientation as Euler XYZ angles or UR rotvec axis-angle.",
    )
    parser.add_argument("--list", action="store_true", help="List replay buffer contents and exit.")
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.60",
        help="UR5 IP address. Only used with --execute.",
    )
    parser.add_argument(
        "--reset-joints",
        type=float,
        nargs=6,
        default=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        help="Joint-space reset pose applied with moveJ before replay starts.",
    )
    parser.add_argument(
        "--reset-settle-seconds",
        type=float,
        default=1.0,
        help="Sleep after reset_joint before starting trajectory replay.",
    )
    parser.add_argument(
        "--gripper-port",
        type=int,
        default=63352,
        help="Robotiq gripper port. Only used with --execute.",
    )
    parser.add_argument(
        "--move-acc",
        type=float,
        default=0.25,
        help="UR moveL acceleration.",
    )
    parser.add_argument(
        "--move-vel",
        type=float,
        default=0.25,
        help="UR moveL velocity.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Open if gripper >= threshold, else close.",
    )
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.1,
        help="Sleep between executed waypoints in seconds.",
    )
    parser.add_argument(
        "--initial-move-seconds",
        type=float,
        default=1.5,
        help="Interpolation time to move from current pose to the first waypoint.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect to the robot and replay the waypoints.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    replay_dir = args.replay_dir.expanduser().resolve()
    pkl_dir = None if args.pkl_dir is None else args.pkl_dir.expanduser().resolve()
    if args.list:
        _list_replay_buffer(replay_dir)
        return

    if pkl_dir is not None:
        print(f"PKL directory: {pkl_dir}")
        waypoints = _load_pkl_waypoints(
            pkl_dir=pkl_dir,
            pkl_key=args.pkl_key,
            action_dim=args.action_dim,
        )
    else:
        trajectory_path = (
            args.trajectory_path.expanduser().resolve()
            if args.trajectory_path is not None
            else _resolve_trajectory_path(replay_dir, args.rank, args.trajectory_id)
        )
        print(f"Trajectory file: {trajectory_path}")
        waypoints = _load_waypoints(
            trajectory_path=trajectory_path,
            batch_index=args.batch_index,
            action_dim=args.action_dim,
        )
    _summarize_waypoints(waypoints)

    if not args.execute:
        print("Dry-run only. Re-run with --execute to command the robot.")
        return

    controller = CustomUR5Controller.launch_controller(
        robot_ip=args.robot_ip,
        env_idx=0,
        node_rank=0,
        worker_rank=0,
        gripper_type="robotiq",
        gripper_port=args.gripper_port,
        tcp_offset=None,
        move_acc=args.move_acc,
        move_vel=args.move_vel,
    )
    try:
        print(f"Resetting robot joints: {np.round(args.reset_joints, 6).tolist()}")
        _reset_robot_joints(
            controller,
            reset_joints=np.asarray(args.reset_joints, dtype=np.float32),
            settle_seconds=args.reset_settle_seconds,
        )
        _execute_waypoints(
            controller,
            waypoints,
            orientation_mode=args.orientation_mode,
            gripper_threshold=args.gripper_threshold,
            initial_move_seconds=args.initial_move_seconds,
            step_sleep=args.step_sleep,
        )
    finally:
        del controller


if __name__ == "__main__":
    main()
