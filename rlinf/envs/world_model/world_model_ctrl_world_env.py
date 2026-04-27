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

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from rlinf.data.datasets.lerobot_book import LeRobotBookTrajectoryDatasetWrapper
from rlinf.data.datasets.lerobot_world_model import LeRobotTrajectoryDatasetWrapper
from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["CtrlWorldEnv"]


@dataclass
class _CtrlWorldArgs:
    """Ctrl-World 构造函数需要的最小参数集合。"""

    svd_model_path: str
    clip_model_path: str
    action_dim: int
    num_history: int
    num_frames: int
    text_cond: bool


class CtrlWorldEnv(BaseWorldEnv):
    """基于 Ctrl-World 的世界模型环境。

    高层流程：
    1. `reset`：从数据集选择初始帧，初始化 latent 与历史状态。
    2. `chunk_step`：输入 `[B, chunk, action_dim]` 的动作块，
       用 Ctrl-World 预测下一段 latent/图像，再用奖励模型给分并返回 chunk 级别的奖励与终止信号。
    3. 在 chunk 之间持续维护内部状态（`current_latent`、`history_latents`、
       `action_history`、episode 指标等）。
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        # Ctrl-World 运行配置。
        self.ctrl_world_cfg = self.cfg.ctrl_world_cfg
        self.inference_dtype = self._to_torch_dtype(
            self.ctrl_world_cfg.get("dtype", "bf16")
        )

        # 分组用于让 reset state id 与 GRPO 的 group 对齐。
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        self.chunk = int(self.ctrl_world_cfg.chunk)
        self.condition_frame_length = int(
            self.ctrl_world_cfg.get("condition_frame_length", 1)
        )
        self.num_history = int(self.ctrl_world_cfg.get("num_history", 6))
        self.action_dim = int(self.ctrl_world_cfg.get("action_dim", 7))
        self.policy_action_type = str(
            self.ctrl_world_cfg.get("policy_action_type", "delta_eef")
        ).lower()
        self.policy_gripper_type = str(
            self.ctrl_world_cfg.get("policy_gripper_type", "passthrough")
        ).lower()

        self.image_size = tuple(self.ctrl_world_cfg.get("image_size", [192, 320]))
        # Ctrl-World 内部以“纵向拼接的 3 视角图像”作为输入。
        self.full_image_size = (self.image_size[0] * 3, self.image_size[1])

        self.main_view_index = int(self.ctrl_world_cfg.get("main_view_index", 1))
        self.wrist_view_index = int(self.ctrl_world_cfg.get("wrist_view_index", 2))
        if self.main_view_index == self.wrist_view_index:
            raise ValueError("main_view_index and wrist_view_index must be different")
        self.extra_view_index = next(
            idx for idx in range(3) if idx not in {self.main_view_index, self.wrist_view_index}
        )

        self.num_inference_steps = int(self.ctrl_world_cfg.get("num_inference_steps", 50))
        self.decode_chunk_size = int(self.ctrl_world_cfg.get("decode_chunk_size", 7))
        self.guidance_scale = float(self.ctrl_world_cfg.get("guidance_scale", 1.0))
        self.fps = int(self.ctrl_world_cfg.get("fps", 7))
        self.motion_bucket_id = int(self.ctrl_world_cfg.get("motion_bucket_id", 127))
        self.frame_level_cond = bool(self.ctrl_world_cfg.get("frame_level_cond", True))
        self.his_cond_zero = bool(self.ctrl_world_cfg.get("his_cond_zero", False))
        self.text_cond = bool(self.ctrl_world_cfg.get("text_cond", True))
        if not (0 <= self.main_view_index <= 2 and 0 <= self.wrist_view_index <= 2):
            raise ValueError("main_view_index and wrist_view_index must be in [0, 2]")

        self.ctrl_world_pipeline_cls = None
        self.model = self._build_ctrl_world_model().eval().to(self.device, self.inference_dtype)
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # 世界模型输入前使用的动作归一化统计量。
        self.action_stats = self._load_action_stats()

        # 数据集图像范围是 [0,1]，Ctrl-World 期望 [-1,1]。
        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        # rollout 过程中的状态缓存。
        self.current_obs = None
        self.current_wrist_obs = None
        self.current_extra_view_obs = None
        self.current_latent = None
        self.history_latents = None
        self.latest_dynamic_gammas = None
        self.current_states = None
        self.action_history = torch.zeros(
            self.num_envs,
            self.num_history,
            self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )

        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        self._is_offloaded = False
        if not torch.is_tensor(self.elapsed_steps):
            self.elapsed_steps = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )

    def _build_dataset(self, cfg):
        """构建 reset-state 数据集。"""
        dataset_type = str(cfg.get("initial_image_dataset_type", "npy")).lower()
        if dataset_type == "lerobot":
            return LeRobotTrajectoryDatasetWrapper(cfg.initial_image_path)
        if dataset_type == "lerobot_book":
            return LeRobotBookTrajectoryDatasetWrapper(cfg.initial_image_path)
        if dataset_type == "npy":
            return NpyTrajectoryDatasetWrapper(
                cfg.initial_image_path, enable_kir=cfg.get("enable_kir", False)
            )
        raise ValueError(f"Unsupported initial_image_dataset_type: {dataset_type}")

    def _to_torch_dtype(self, dtype: str) -> torch.dtype:
        """将配置中的 dtype 字符串映射为 torch dtype。"""
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        key = str(dtype).lower()
        if key not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dtype_map[key]

    def _import_ctrl_world_modules(self):
        """从外部仓库路径动态导入 Ctrl-World 模块。"""
        ctrl_world_repo_path = self.ctrl_world_cfg.get("ctrl_world_repo_path", None)
        if ctrl_world_repo_path is None:
            ctrl_world_repo_path = os.environ.get("CTRL_WORLD_PATH", None)

        if ctrl_world_repo_path is None:
            raise ValueError(
                "ctrl_world_repo_path is required in env.ctrl_world_cfg or CTRL_WORLD_PATH env variable"
            )

        ctrl_world_repo_path = str(Path(ctrl_world_repo_path).expanduser().resolve())
        if not os.path.isdir(ctrl_world_repo_path):
            raise ValueError(f"ctrl_world_repo_path does not exist: {ctrl_world_repo_path}")

        if ctrl_world_repo_path not in sys.path:
            sys.path.insert(0, ctrl_world_repo_path)

        try:
            from models.ctrl_world import CrtlWorld  # type: ignore
            from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import Ctrl-World modules. Please ensure Ctrl-World dependencies "
                "are installed and ctrl_world_repo_path points to the repository root."
            ) from e

        return CrtlWorld, CtrlWorldDiffusionPipeline

    def _build_ctrl_world_model(self):
        """实例化 Ctrl-World 模型并加载 checkpoint 权重。"""
        CrtlWorld, CtrlWorldDiffusionPipeline = self._import_ctrl_world_modules()
        self.ctrl_world_pipeline_cls = CtrlWorldDiffusionPipeline

        args = _CtrlWorldArgs(
            svd_model_path=self.ctrl_world_cfg.svd_model_path,
            clip_model_path=self.ctrl_world_cfg.clip_model_path,
            action_dim=self.action_dim,
            num_history=self.num_history,
            num_frames=self.chunk,
            text_cond=self.text_cond,
        )

        model = CrtlWorld(args)

        ckpt_path = self.ctrl_world_cfg.ckpt_path
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Ctrl-World checkpoint path does not exist: {ckpt_path}")

        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(raw_ckpt)

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # 兼容被 accelerator/DDP 封装过的 checkpoint。
            model.load_state_dict(state_dict, strict=False)

        return model

    def _extract_state_dict(self, ckpt_obj):
        """从不同封装格式的 checkpoint 中提取纯 state_dict。"""
        state_dict = ckpt_obj
        if isinstance(state_dict, dict):
            for key in ["state_dict", "model_state_dict", "model", "module"]:
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

        if not isinstance(state_dict, dict):
            raise ValueError("Invalid Ctrl-World checkpoint format.")

        if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        return state_dict

    def _load_reward_model(self):
        """加载 `ctrl_world_cfg.reward_model` 指定的帧级奖励模型。"""
        reward_cfg = self.ctrl_world_cfg.get("reward_model", self.cfg.get("reward_model", None))
        if reward_cfg is None:
            raise ValueError("reward_model config is required for CtrlWorldEnv")

        if reward_cfg.type == "ResnetRewModel":
            from diffsynth.models.reward_model import ResnetRewModel

            rew_model = ResnetRewModel(reward_cfg.from_pretrained)
        elif reward_cfg.type == "TaskEmbedResnetRewModel":
            from diffsynth.models.reward_model import TaskEmbedResnetRewModel

            rew_model = TaskEmbedResnetRewModel(
                checkpoint_path=reward_cfg.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        else:
            raise ValueError(f"Unknown reward model type: {reward_cfg.type}")

        return rew_model

    def _load_action_stats(self):
        """加载动作分位数统计，用于归一化到 [-1,1]。"""
        stats_path = self.ctrl_world_cfg.get("data_stat_path", None)
        if stats_path is None or not os.path.exists(stats_path):
            raise ValueError(f"Ctrl-World data_stat_path does not exist: {stats_path}")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        if "action_01" not in stats or "action_99" not in stats:
            raise ValueError(
                f"Expected 'action_01' and 'action_99' in {stats_path}, got keys: {list(stats.keys())}"
            )

        q01 = np.asarray(stats["action_01"], dtype=np.float32)
        q99 = np.asarray(stats["action_99"], dtype=np.float32)
        return {"q01": q01, "q99": q99}

    def _normalize_action(self, actions: np.ndarray) -> np.ndarray:
        """用数据集 q01/q99 对原始动作做归一化并裁剪到 [-1,1]。"""
        q01 = self.action_stats["q01"]
        q99 = self.action_stats["q99"]

        action_dim = min(actions.shape[-1], q01.shape[-1], q99.shape[-1])
        actions_norm = actions.copy()
        actions_norm[..., :action_dim] = (
            2
            * (
                (actions_norm[..., :action_dim] - q01[:action_dim])
                / (q99[:action_dim] - q01[:action_dim] + 1e-8)
            )
            - 1
        )
        return np.clip(actions_norm, -1.0, 1.0)

    def _convert_policy_actions_to_ctrl_world(self, actions_np: np.ndarray) -> np.ndarray:
        """Convert policy outputs to the action format expected by Ctrl-World."""
        if self.policy_action_type not in {"delta_eef", "absolute_eef"}:
            raise ValueError(
                f"Unsupported policy_action_type: {self.policy_action_type}. "
                "Expected one of {'delta_eef', 'absolute_eef'}."
            )

        if self.action_dim < 7:
            raise ValueError(
                f"{self.policy_action_type} conversion expects action_dim >= 7, got {self.action_dim}"
            )

        return actions_np.astype(np.float32, copy=True)

    def _build_initial_action_history(
        self, current_states: torch.Tensor, num_reset_envs: int
    ) -> torch.Tensor:
        """Initialize action history in the same action convention as Ctrl-World training."""
        init_actions = torch.zeros(
            (num_reset_envs, self.action_dim), device=self.device, dtype=torch.float32
        )

        if self.policy_action_type == "absolute_eef":
            state_action_dim = min(current_states.shape[1], self.action_dim)
            init_actions[:, :state_action_dim] = current_states[:, :state_action_dim]

        return init_actions.unsqueeze(1).repeat(1, self.num_history, 1)

    def _update_current_states(self, last_action: torch.Tensor, num_envs: int) -> None:
        """Update cached policy states according to the configured action convention."""
        if self.current_states is None:
            state_dim = max(self.action_dim, 8 if self.policy_action_type == "delta_eef" else 7)
            self.current_states = torch.zeros(
                (num_envs, state_dim), device=self.device, dtype=torch.float32
            )

        pose_dim = min(6, self.current_states.shape[1], last_action.shape[1])
        if self.policy_action_type == "delta_eef":
            self.current_states[:, :pose_dim] = (
                self.current_states[:, :pose_dim] + last_action[:, :pose_dim]
            )
        elif self.policy_action_type == "absolute_eef":
            self.current_states[:, :pose_dim] = last_action[:, :pose_dim]
        else:
            raise ValueError(f"Unsupported policy_action_type: {self.policy_action_type}")

        if self.current_states.shape[1] >= 8 and last_action.shape[1] >= 7:
            if self.policy_action_type == "absolute_eef":
                next_finger_pos = last_action[:, 6:7]
            else:
                gripper_state_scale = self.current_states[:, 6:8].abs().mean(
                    dim=1, keepdim=True
                )
                gripper_state_scale = torch.clamp(
                    gripper_state_scale, min=0.04
                ).to(last_action.dtype)
                gripper_cmd = torch.sign(torch.clamp(last_action[:, 6:7], -1.0, 1.0))
                next_finger_pos = torch.where(
                    gripper_cmd > 0,
                    gripper_state_scale,
                    torch.where(
                        gripper_cmd < 0,
                        -gripper_state_scale,
                        self.current_states[:, 6:7],
                    ),
                )

            self.current_states[:, 6:7] = next_finger_pos
            self.current_states[:, 7:8] = -next_finger_pos
        elif self.current_states.shape[1] >= 7 and last_action.shape[1] >= 7:
            if self.policy_action_type == "absolute_eef":
                self.current_states[:, 6:7] = last_action[:, 6:7]
            else:
                self.current_states[:, 6:7] = torch.sign(
                    torch.clamp(last_action[:, 6:7], -1.0, 1.0)
                )

    def _init_metrics(self):
        self.elapsed_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.elapsed_steps[mask] = 0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.elapsed_steps[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor

        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.to(torch.float32).clone()
        episode_info["reward"] = episode_info["return"] / torch.clamp(
            episode_info["episode_len"], min=1.0
        )
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        """根据配置将绝对奖励转换为相对奖励。"""
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        """通过 chunk 内最大帧奖励近似估计是否成功。"""
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]
        success_estimated = max_reward_in_chunk >= success_threshold
        return success_estimated.to(self.device)

    def _compute_dynamic_gammas(
        self,
        pred_latents: torch.Tensor,
        prev_current_latent: Optional[torch.Tensor],
        prev_history_latents: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Build per-step dynamic gammas from final Ctrl-World latents."""
        if not bool(self.cfg.get("dynamic_gamma_enabled", False)):
            return None
        if prev_current_latent is None or prev_history_latents is None:
            raise ValueError(
                "dynamic gamma requires current_latent and history_latents to be initialized"
            )

        gamma_base = float(self.cfg.get("dynamic_gamma_base", 0.98))
        gamma_min = float(self.cfg.get("dynamic_gamma_min", 0.9))
        gamma_max = float(self.cfg.get("dynamic_gamma_max", 1.0))
        delta_weight = float(self.cfg.get("dynamic_gamma_delta_weight", 0.01))
        gap_weight = float(self.cfg.get("dynamic_gamma_gap_weight", 0.02))
        norm_eps = 1e-6

        pred_latents = pred_latents.detach().to(torch.float32)
        prev_current_latent = prev_current_latent.detach().to(torch.float32)
        prev_history_latents = prev_history_latents.detach().to(torch.float32)

        prev_step_latents = torch.cat(
            [prev_current_latent.unsqueeze(1), pred_latents[:, :-1]], dim=1
        )
        latent_delta_norm = (pred_latents - prev_step_latents).pow(2).mean(
            dim=(2, 3, 4)
        ).sqrt()

        history_anchor = prev_history_latents[:, -1].unsqueeze(1).expand_as(pred_latents)
        history_gap_norm = (pred_latents - history_anchor).pow(2).mean(
            dim=(2, 3, 4)
        ).sqrt()

        latent_delta_norm = latent_delta_norm / latent_delta_norm.mean(
            dim=1, keepdim=True
        ).clamp_min(norm_eps)
        history_gap_norm = history_gap_norm / history_gap_norm.mean(
            dim=1, keepdim=True
        ).clamp_min(norm_eps)

        # Center normalized features so gamma_base remains the default operating point.
        score_t = delta_weight * (latent_delta_norm - 1.0) - gap_weight * (
            history_gap_norm - 1.0
        )
        dynamic_gammas = torch.clamp(
            gamma_base + score_t,
            min=gamma_min,
            max=gamma_max,
        )
        return dynamic_gammas.to(torch.float32)

    def update_reset_state_ids(self):
        """每个 group 采样一个 reset episode id，并复制给组内成员。"""
        total_num_episodes = len(self.dataset)
        reset_state_ids = torch.randint(
            low=0,
            high=total_num_episodes,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _encode_full_image_to_latent(self, full_images: torch.Tensor) -> torch.Tensor:
        """将完整（3 视角拼接）图像编码到 VAE latent。"""
        with torch.no_grad():
            latents = (
                self.model.pipeline.vae.encode(full_images)
                .latent_dist.sample()
                .mul_(self.model.pipeline.vae.config.scaling_factor)
            )
        return latents

    def _decode_latents_to_views(self, latents: torch.Tensor):
        """将 latent 视频解码，并拆分成纵向拼接的 3 个视角。"""
        # latents 形状: [B, T, 4, H', W']
        bsz, t = latents.shape[:2]
        flat_latents = latents.flatten(0, 1)

        decoded_videos = []
        decode_kwargs = {}
        for i in range(0, flat_latents.shape[0], self.decode_chunk_size):
            chunk = (
                flat_latents[i : i + self.decode_chunk_size]
                / self.model.pipeline.vae.config.scaling_factor
            )
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_videos.append(self.model.pipeline.vae.decode(chunk, **decode_kwargs).sample)

        videos = torch.cat(decoded_videos, dim=0)
        videos = videos.reshape(bsz, t, *videos.shape[1:])  # [B, T, 3, H, W]

        # 将纵向拼接图像拆为 3 个视角。
        views = torch.chunk(videos, chunks=3, dim=3)
        if len(views) != 3:
            raise ValueError(
                f"Expected 3 views after split, got {len(views)} with video shape {videos.shape}"
            )
        return views

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        env_idx: Optional[Union[list[int], np.ndarray, torch.Tensor]] = None,
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """从数据集起始帧重置环境状态。

        初始化内容包括：
        - 当前图像观测
        - 当前 latent 与 latent 历史
        - 动作历史（若有初始末端位姿则用其初始化）
        - 文本条件所需的任务描述
        """
        self.onload()

        if env_idx is None:
            target_env_idx = torch.arange(self.num_envs, device=self.device)
        else:
            target_env_idx = torch.as_tensor(
                env_idx, device=self.device, dtype=torch.long
            ).reshape(-1)
            if target_env_idx.numel() == 0:
                return self._wrap_obs(), {}

        if self.is_start and env_idx is None:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False
        self.latest_dynamic_gammas = None

        num_reset_envs = int(target_env_idx.numel())
        if len(self.dataset) < num_reset_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {num_reset_envs}"
            )

        if episode_indices is None:
            if seed is not None:
                if isinstance(seed, list):
                    np.random.seed(seed[0])
                else:
                    np.random.seed(seed)
            episode_indices = np.random.choice(
                len(self.dataset), size=num_reset_envs, replace=False
            )
        else:
            if isinstance(episode_indices, torch.Tensor):
                episode_indices = episode_indices.cpu().numpy()
            episode_indices = np.asarray(episode_indices)
            if episode_indices.shape[0] != num_reset_envs:
                raise ValueError(
                    f"Expected {num_reset_envs} episode indices, got {episode_indices.shape[0]}"
                )

        main_imgs = []
        wrist_imgs = []
        extra_view_imgs = []
        full_imgs = []
        task_descriptions = []
        init_ee_poses = []

        for episode_idx in episode_indices:
            episode_data = self.dataset[int(episode_idx)]
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]
            task_desc = str(episode_data.get("task", ""))
            task_descriptions.append(task_desc)

            if "image" not in first_frame:
                raise ValueError(f"No 'image' key in frame for episode {episode_idx}")

            default_image = first_frame["image"]
            view_tensors = [
                first_frame.get("main_image_1", default_image),
                first_frame.get("main_image_2", default_image),
                first_frame.get("wrist_image", default_image),
            ]

            normalized_views = []
            for view_tensor in view_tensors:
                if view_tensor.shape[1:] != self.image_size:
                    view_tensor = F.interpolate(
                        view_tensor.unsqueeze(0),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                normalized_views.append(self.trans_norm(view_tensor))

            full_img = torch.cat(normalized_views, dim=1)
            selected_main_view = normalized_views[self.main_view_index]
            selected_wrist_view = normalized_views[self.wrist_view_index]
            selected_extra_view = normalized_views[self.extra_view_index]

            main_imgs.append(selected_main_view)
            wrist_imgs.append(selected_wrist_view)
            extra_view_imgs.append(selected_extra_view)
            full_imgs.append(full_img)

            if "observation.state" in first_frame:
                init_pose = first_frame["observation.state"].detach().cpu().numpy()
                init_ee_poses.append(init_pose)
            else:
                init_ee_poses.append(None)

        stacked_main = torch.stack(main_imgs, dim=0).to(self.device)  # [B,3,H,W]
        stacked_main = stacked_main.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, 1, self.condition_frame_length, 1, 1
        )
        stacked_wrist = torch.stack(wrist_imgs, dim=0).to(self.device)
        stacked_wrist = stacked_wrist.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, 1, self.condition_frame_length, 1, 1
        )
        stacked_extra_view = torch.stack(extra_view_imgs, dim=0).to(self.device)
        stacked_extra_view = stacked_extra_view.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, 1, self.condition_frame_length, 1, 1
        )

        full_images = torch.stack(full_imgs, dim=0).to(self.device, self.inference_dtype)
        current_latent = self._encode_full_image_to_latent(full_images)

        history_latents = current_latent.unsqueeze(1).repeat(
            1, self.num_history, 1, 1, 1
        )

        init_states = []
        state_dim = 0
        for init_ee_pose in init_ee_poses:
            if init_ee_pose is None:
                init_state = np.zeros(7, dtype=np.float32)
            else:
                init_state = np.asarray(init_ee_pose, dtype=np.float32).reshape(-1)
                if init_state.shape[0] < 7:
                    init_state = np.pad(init_state, (0, 7 - init_state.shape[0]))
            state_dim = max(state_dim, init_state.shape[0])
            init_states.append(init_state)
        state_dim = max(state_dim, 7)
        if self.policy_action_type == "delta_eef":
            state_dim = max(state_dim, 8)
        init_states = [
            np.pad(state, (0, state_dim - state.shape[0])) if state.shape[0] < state_dim else state[:state_dim]
            for state in init_states
        ]
        current_states = torch.from_numpy(np.stack(init_states, axis=0)).to(
            self.device, torch.float32
        )
        action_history = self._build_initial_action_history(current_states, num_reset_envs)

        if env_idx is None or self.current_obs is None:
            self.current_obs = stacked_main
            self.current_wrist_obs = stacked_wrist
            self.current_extra_view_obs = stacked_extra_view
            self.current_latent = current_latent
            self.history_latents = history_latents
            self.current_states = current_states
            self.action_history = action_history
            self.task_descriptions = task_descriptions
            self.init_ee_poses = init_ee_poses
        else:
            self.current_obs[target_env_idx] = stacked_main
            self.current_wrist_obs[target_env_idx] = stacked_wrist
            self.current_extra_view_obs[target_env_idx] = stacked_extra_view
            self.current_latent[target_env_idx] = current_latent
            self.history_latents[target_env_idx] = history_latents
            self.current_states[target_env_idx] = current_states
            self.action_history[target_env_idx] = action_history
            target_env_idx_list = target_env_idx.tolist()
            for local_i, global_i in enumerate(target_env_idx_list):
                self.task_descriptions[global_i] = task_descriptions[local_i]
                self.init_ee_poses[global_i] = init_ee_poses[local_i]

        self._reset_metrics(target_env_idx)

        extracted_obs = self._wrap_obs()
        infos = {}
        return extracted_obs, infos

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        """Ctrl-World 环境不支持单步 step，仅支持 chunk_step。"""
        raise NotImplementedError(
            "step in CtrlWorldEnv is not implemented, use chunk_step instead"
        )

    def _infer_next_chunk_rewards(self):
        """对最新生成的 chunk 帧运行奖励模型。"""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        # [B, C, V, T, H, W] -> 取最新 chunk -> [B*chunk, C, H, W]
        extract_chunk_obs = self.current_obs.permute(0, 3, 1, 2, 4, 5)
        extract_chunk_obs = extract_chunk_obs[:, -self.chunk :, :, :, :, :]
        extract_chunk_obs = extract_chunk_obs.reshape(self.num_envs * self.chunk, 3, 1, *self.image_size)
        extract_chunk_obs = extract_chunk_obs.squeeze(2).to(self.device)
        # Align online reward inference with reward-model training preprocessing.
        extract_chunk_obs = F.interpolate(
            extract_chunk_obs,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        # return torch.zeros((self.num_envs, self.chunk), dtype=torch.float32, device=self.device)

        reward_cfg = self.ctrl_world_cfg.get("reward_model", self.cfg.get("reward_model", None))
        if reward_cfg.type == "ResnetRewModel":
            rewards = self.reward_model.predict_rew(extract_chunk_obs)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        elif reward_cfg.type == "TaskEmbedResnetRewModel":
            instructions = []
            for env_idx in range(self.num_envs):
                instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
            rewards = self.reward_model.predict_rew(extract_chunk_obs, instructions)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        else:
            raise ValueError(f"Unknown reward model type: {reward_cfg.type}")

        return rewards

    def _infer_next_chunk_frames(self, actions):
        """在动作历史条件下生成下一段 chunk 的 latent/视频。"""
        num_envs = self.num_envs
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        if actions_np.shape != (num_envs, self.chunk, self.action_dim):
            raise ValueError(
                f"Expected actions shape {(num_envs, self.chunk, self.action_dim)}, got {actions_np.shape}"
            )

        ctrl_world_actions_np = self._convert_policy_actions_to_ctrl_world(actions_np)
        action_cond_raw = np.concatenate(
            [self.action_history.detach().cpu().numpy(), ctrl_world_actions_np], axis=1
        )
        action_cond_norm = self._normalize_action(action_cond_raw)
        action_cond = (
            torch.from_numpy(action_cond_norm)
            .to(self.device)
            .to(self.inference_dtype)
        )

        prev_current_latent = self.current_latent
        prev_history_latents = self.history_latents

        # 构造文本/动作条件 token，并在 latent 空间执行扩散推理。
        with torch.no_grad():
            if self.text_cond:
                text_token = self.model.action_encoder(
                    action_cond,
                    self.task_descriptions,
                    self.model.tokenizer,
                    self.model.text_encoder,
                )
            else:
                text_token = self.model.action_encoder(action_cond)

            _, pred_latents = self.ctrl_world_pipeline_cls.__call__(
                self.model.pipeline,
                image=self.current_latent.to(self.inference_dtype),
                text=text_token,
                width=self.image_size[1],
                height=self.full_image_size[0],
                num_frames=self.chunk,
                history=self.history_latents.to(self.inference_dtype),
                num_inference_steps=self.num_inference_steps,
                decode_chunk_size=self.decode_chunk_size,
                max_guidance_scale=self.guidance_scale,
                fps=self.fps,
                motion_bucket_id=self.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=self.frame_level_cond,
                his_cond_zero=self.his_cond_zero,
            )

        pred_latents = pred_latents.to(self.device, self.inference_dtype)
        self.latest_dynamic_gammas = self._compute_dynamic_gammas(
            pred_latents,
            prev_current_latent,
            prev_history_latents,
        )

        # 仅保留最后一帧 latent 作为当前状态，并滑动更新历史窗口。
        self.current_latent = pred_latents[:, -1].detach()
        self.history_latents = torch.cat(
            [self.history_latents[:, 1:], self.current_latent.unsqueeze(1)], dim=1
        )

        # 追加最新动作并滑动更新动作历史窗口。
        last_action = torch.from_numpy(ctrl_world_actions_np[:, -1, :]).to(
            self.device, torch.float32
        )
        self.action_history = torch.cat(
            [self.action_history[:, 1:], last_action.unsqueeze(1)], dim=1
        )
        self._update_current_states(last_action, num_envs)

        # 解码生成的 chunk，并追加到观测历史中。
        views = self._decode_latents_to_views(pred_latents)

        main_video = views[self.main_view_index].permute(0, 2, 1, 3, 4)  # [B,3,T,H,W]
        wrist_video = views[self.wrist_view_index].permute(0, 2, 1, 3, 4)
        extra_view_video = views[self.extra_view_index].permute(0, 2, 1, 3, 4)

        x_main = main_video.unsqueeze(2)   # [B,3,1,T,H,W]
        x_wrist = wrist_video.unsqueeze(2) # [B,3,1,T,H,W]
        x_extra = extra_view_video.unsqueeze(2)  # [B,3,1,T,H,W]

        self.current_obs = torch.cat([self.current_obs, x_main], dim=3)
        self.current_wrist_obs = torch.cat([self.current_wrist_obs, x_wrist], dim=3)
        self.current_extra_view_obs = torch.cat(
            [self.current_extra_view_obs, x_extra], dim=3
        )

        max_frames = self.condition_frame_length + self.chunk * 2
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]
            self.current_wrist_obs = self.current_wrist_obs[:, :, :, -max_frames:, :, :]
            self.current_extra_view_obs = self.current_extra_view_obs[
                :, :, :, -max_frames:, :, :
            ]

    def _wrap_obs(self):
        """将内部归一化张量转换为策略侧使用的 uint8 观测。"""
        num_envs = self.num_envs

        b, c, v, t, h, w = self.current_obs.shape
        assert b == num_envs

        main_last = self.current_obs[:, :, 0, -1, :, :]   # [B,3,H,W]
        wrist_last = self.current_wrist_obs[:, :, 0, -1, :, :] if self.current_wrist_obs is not None else None

        main_image = main_last.permute(0, 2, 3, 1)
        main_image = (main_image + 1.0) / 2.0 * 255.0
        main_image = torch.clamp(main_image, 0, 255).to(torch.uint8)

        wrist_image = None
        if wrist_last is not None:
            wrist_image = wrist_last.permute(0, 2, 3, 1)
            wrist_image = (wrist_image + 1.0) / 2.0 * 255.0
            wrist_image = torch.clamp(wrist_image, 0, 255).to(torch.uint8)

        extra_view_image = None
        if self.current_extra_view_obs is not None:
            extra_view_last = self.current_extra_view_obs[:, :, 0, -1, :, :]
            extra_view_image = extra_view_last.permute(0, 2, 3, 1)
            extra_view_image = (extra_view_image + 1.0) / 2.0 * 255.0
            extra_view_image = torch.clamp(extra_view_image, 0, 255).to(torch.uint8)

        if self.current_states is None:
            states = torch.zeros((num_envs, 8), device=self.device, dtype=torch.float32)
        else:
            states = self.current_states.to(device=self.device, dtype=torch.float32)

        obs = {
            "main_images": main_image,
            "wrist_images": wrist_image,
            "extra_view_images": extra_view_image,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _get_latest_chunk_main_images(self) -> torch.Tensor:
        """Return the latest chunk of main-view images as uint8 `[B, chunk, H, W, C]`."""
        latest_chunk = self.current_obs[:, :, 0, -self.chunk :, :, :]  # [B,3,T,H,W]
        latest_chunk = latest_chunk.permute(0, 2, 3, 4, 1)  # [B,T,H,W,C]
        latest_chunk = (latest_chunk + 1.0) / 2.0 * 255.0
        return torch.clamp(latest_chunk, 0, 255).to(torch.uint8)

    def _get_latest_chunk_wrist_images(self) -> Optional[torch.Tensor]:
        """Return the latest chunk of wrist-view images as uint8 `[B, chunk, H, W, C]`."""
        if self.current_wrist_obs is None:
            return None
        latest_chunk = self.current_wrist_obs[:, :, 0, -self.chunk :, :, :]  # [B,3,T,H,W]
        latest_chunk = latest_chunk.permute(0, 2, 3, 4, 1)  # [B,T,H,W,C]
        latest_chunk = (latest_chunk + 1.0) / 2.0 * 255.0
        return torch.clamp(latest_chunk, 0, 255).to(torch.uint8)

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """对 done 环境执行自动重置，并保留终止前的观测/信息。"""
        final_obs = extracted_obs
        final_info = infos
        done_env_idx = torch.where(dones)[0]
        extracted_obs, infos = self.reset(env_idx=done_env_idx)
        for key in (
            "chunk_raw_rewards",
            "dynamic_gammas",
            "success_frame_time_idx",
            "success_frame_images",
            "success_frame_raw_images",
            "success_frame_raw_tensors",
            "success_frame_meta",
            "success_frame_wrist_images",
        ):
            if key in final_info:
                infos[key] = final_info[key]

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones

        return extracted_obs, infos

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """RL 主循环使用的环境步进接口（按动作块）。

        Args:
            policy_output_action: 形状为 `[B, chunk, action_dim]` 的 Tensor/ndarray。

        Returns:
            向量环境 chunk 接口风格的元组：
            `([obs], rewards, terminations, truncations, [infos])`。
        """
        self.onload()
        if isinstance(policy_output_action, torch.Tensor):
            policy_actions_for_video = policy_output_action.detach().to(
                device=self.device, dtype=torch.float32
            )
        else:
            policy_actions_for_video = torch.as_tensor(
                policy_output_action, device=self.device, dtype=torch.float32
            )

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._infer_next_chunk_frames(policy_output_action)

        # 这里的一次环境步进对应整个 chunk，而不是单帧。
        self.elapsed_steps += self.chunk

        extracted_obs = self._wrap_obs()

        # 先对最新生成的 chunk 逐帧打分，再转换成训练使用的奖励。
        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)#计算差分奖励

        # 成功判定和训练奖励累计是两条独立逻辑。
        estimated_success = self._estimate_success_from_rewards(chunk_rewards)

        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        # 为兼容向量环境接口，只在 chunk 的最后一帧位置标记终止。
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        success_frame_time_idx = chunk_rewards.argmax(dim=1)
        latest_chunk_main_tensors = self.current_obs[:, :, 0, -self.chunk :, :, :]
        latest_chunk_main_tensors = latest_chunk_main_tensors.permute(0, 2, 1, 3, 4).contiguous()
        latest_chunk_main_images = self._get_latest_chunk_main_images()
        latest_chunk_wrist_images = self._get_latest_chunk_wrist_images()
        success_frame_images = latest_chunk_main_images[
            torch.arange(self.num_envs, device=self.device), success_frame_time_idx
        ]
        success_frame_raw_tensors = latest_chunk_main_tensors[
            torch.arange(self.num_envs, device=self.device), success_frame_time_idx
        ].detach().to(torch.float32).cpu()
        success_frame_raw_images = success_frame_raw_tensors.permute(0, 2, 3, 1)
        success_frame_raw_images = torch.clamp(
            (success_frame_raw_images + 1.0) / 2.0 * 255.0, 0, 255
        ).to(torch.uint8)
        success_frame_meta = []
        success_threshold = float(getattr(self.cfg, "success_reward_threshold", 0.9))
        chunk_rewards_cpu = chunk_rewards.detach().to(torch.float32).cpu()
        success_frame_time_idx_cpu = success_frame_time_idx.detach().cpu()
        estimated_success_cpu = estimated_success.detach().cpu()
        elapsed_steps_cpu = self.elapsed_steps.detach().cpu()
        for env_id in range(self.num_envs):
            frame_idx = int(success_frame_time_idx_cpu[env_id].item())
            chunk_rewards_env = chunk_rewards_cpu[env_id]
            success_frame_meta.append(
                {
                    "env_id": env_id,
                    "elapsed_steps": int(elapsed_steps_cpu[env_id].item()),
                    "success_frame_time_idx": frame_idx,
                    "selected_frame_reward": float(chunk_rewards_env[frame_idx].item()),
                    "max_reward_in_chunk": float(chunk_rewards_env.max().item()),
                    "chunk_raw_rewards": [
                        float(reward_value) for reward_value in chunk_rewards_env.tolist()
                    ],
                    "success_reward_threshold": success_threshold,
                    "success_estimated": bool(estimated_success_cpu[env_id].item()),
                }
            )
        success_frame_wrist_images = None
        if latest_chunk_wrist_images is not None:
            success_frame_wrist_images = latest_chunk_wrist_images[
                torch.arange(self.num_envs, device=self.device), success_frame_time_idx
            ]

        # 当前 chunk 对轨迹回报的贡献，是这个 chunk 内所有帧奖励之和。
        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, {}
        )
        infos["policy_action"] = policy_actions_for_video
        infos["policy_action_last"] = policy_actions_for_video[:, -1, :]
        infos["chunk_raw_rewards"] = chunk_rewards
        infos["dynamic_gammas"] = self.latest_dynamic_gammas
        infos["success_frame_time_idx"] = success_frame_time_idx
        infos["success_frame_images"] = success_frame_images
        infos["success_frame_raw_images"] = success_frame_raw_images
        infos["success_frame_raw_tensors"] = success_frame_raw_tensors
        infos["success_frame_meta"] = success_frame_meta
        infos["success_frame_wrist_images"] = success_frame_wrist_images

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(past_dones, extracted_obs, infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            [extracted_obs],
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    def offload(self):
        """将大模型与状态迁移到 CPU，降低显存占用。"""
        if self._is_offloaded:
            return

        self.model = self.model.to("cpu")
        self.reward_model = self.reward_model.to("cpu")

        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.current_wrist_obs = recursive_to_device(self.current_wrist_obs, "cpu")
        self.current_latent = recursive_to_device(self.current_latent, "cpu")
        self.history_latents = recursive_to_device(self.history_latents, "cpu")
        self.latest_dynamic_gammas = recursive_to_device(
            self.latest_dynamic_gammas, "cpu"
        )
        self.current_states = recursive_to_device(self.current_states, "cpu")
        self.action_history = self.action_history.cpu()

        self.elapsed_steps = self.elapsed_steps.cpu()
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()

        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        """在推理/步进前将状态与模型迁回设备。"""
        if not self._is_offloaded:
            return

        self.model = self.model.to(self.device, self.inference_dtype)
        self.reward_model = self.reward_model.to(self.device)

        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.current_wrist_obs = recursive_to_device(self.current_wrist_obs, self.device)
        self.current_latent = recursive_to_device(self.current_latent, self.device)
        self.history_latents = recursive_to_device(self.history_latents, self.device)
        self.latest_dynamic_gammas = recursive_to_device(
            self.latest_dynamic_gammas, self.device
        )
        self.current_states = recursive_to_device(self.current_states, self.device)
        self.action_history = self.action_history.to(self.device)

        self.elapsed_steps = self.elapsed_steps.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)

        self._is_offloaded = False

    def get_state(self) -> bytes:
        """序列化环境状态，用于 checkpoint 或迁移。"""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "current_wrist_obs": recursive_to_device(self.current_wrist_obs, "cpu")
            if self.current_wrist_obs is not None
            else None,
            "current_latent": recursive_to_device(self.current_latent, "cpu")
            if self.current_latent is not None
            else None,
            "history_latents": recursive_to_device(self.history_latents, "cpu")
            if self.history_latents is not None
            else None,
            "current_states": recursive_to_device(self.current_states, "cpu")
            if self.current_states is not None
            else None,
            "action_history": self.action_history.cpu(),
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "elapsed_steps": self.elapsed_steps.cpu(),
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )

        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """恢复由 `get_state` 生成的序列化环境状态。"""
        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        self.current_obs = (
            recursive_to_device(state["current_obs"], self.device)
            if state["current_obs"] is not None
            else None
        )
        self.current_wrist_obs = (
            recursive_to_device(state["current_wrist_obs"], self.device)
            if state["current_wrist_obs"] is not None
            else None
        )
        self.current_latent = (
            recursive_to_device(state["current_latent"], self.device)
            if state["current_latent"] is not None
            else None
        )
        self.history_latents = (
            recursive_to_device(state["history_latents"], self.device)
            if state["history_latents"] is not None
            else None
        )
        self.current_states = (
            recursive_to_device(state["current_states"], self.device)
            if state.get("current_states") is not None
            else None
        )
        self.action_history = state["action_history"].to(self.device)
        self.task_descriptions = state["task_descriptions"]
        self.init_ee_poses = state["init_ee_poses"]
        self.elapsed_steps = state["elapsed_steps"].to(self.device)
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self._is_start = state["_is_start"]
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])

        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.returns = state["returns"].to(self.device)
