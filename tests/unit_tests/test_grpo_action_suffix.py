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

import torch

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns


def _manual_grpo_action_suffix_advantages(
    rewards: torch.Tensor,
    dynamic_gammas: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    num_chunk, batch_size, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size

    flat_rewards = rewards.transpose(1, 2).reshape(n_steps, batch_size)
    flat_dynamic_gammas = dynamic_gammas.transpose(1, 2).reshape(n_steps, batch_size)
    flat_loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, batch_size)
    flat_dones = dones.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, batch_size)
    flat_dones = flat_dones[-(n_steps + 1) :]

    returns = torch.zeros_like(flat_rewards)
    running = torch.zeros(batch_size, dtype=flat_rewards.dtype)
    for step in reversed(range(n_steps)):
        running = flat_rewards[step] + flat_dynamic_gammas[step] * running * (
            ~flat_dones[step + 1]
        ).to(flat_rewards.dtype)
        returns[step] = running

    advantages = torch.zeros_like(returns)
    num_groups = batch_size // group_size
    for step in range(n_steps):
        grouped_returns = returns[step].view(num_groups, group_size)
        grouped_mean = grouped_returns.mean(dim=-1, keepdim=True)
        grouped_std = grouped_returns.std(dim=-1, keepdim=True)
        step_advantages = (grouped_returns - grouped_mean) / (grouped_std + 1e-6)
        advantages[step] = step_advantages.view(-1)

    advantages = advantages * flat_loss_mask.to(advantages.dtype)
    return advantages.reshape(num_chunk, chunk_size, batch_size).transpose(1, 2)


def test_grpo_action_suffix_matches_manual_dynamic_gamma_returns():
    rewards = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]],
            [[19.0, 23.0], [29.0, 31.0], [37.0, 41.0], [43.0, 47.0]],
        ],
        dtype=torch.float32,
    )
    dynamic_gammas = torch.tensor(
        [
            [[0.95, 0.90], [0.97, 0.92], [0.99, 0.94], [1.00, 0.96]],
            [[0.91, 0.93], [0.92, 0.94], [0.93, 0.95], [0.94, 0.96]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(3, 4, 2, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo_action_suffix",
        rewards=rewards,
        dynamic_gammas=dynamic_gammas,
        dones=dones,
        group_size=2,
        reward_type="action_level",
        loss_mask=loss_mask,
    )

    expected_advantages = _manual_grpo_action_suffix_advantages(
        rewards=rewards,
        dynamic_gammas=dynamic_gammas,
        dones=dones,
        loss_mask=loss_mask,
        group_size=2,
    )

    assert "returns" not in result
    assert result["advantages"].shape == rewards.shape
    assert torch.allclose(result["advantages"], expected_advantages, atol=1e-5)


def test_grpo_action_suffix_gamma_one_matches_suffix_and_respects_masks():
    rewards = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]],
        dtype=torch.float32,
    )
    dynamic_gammas = torch.ones_like(rewards)
    dones = torch.zeros(3, 2, 2, dtype=torch.bool)
    dones[1, 0, 0] = True
    dones[-1] = True
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)
    loss_mask[0, 1, 1] = False

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo_action_suffix",
        rewards=rewards,
        dynamic_gammas=dynamic_gammas,
        dones=dones,
        group_size=2,
        reward_type="action_level",
        loss_mask=loss_mask,
    )

    expected_advantages = _manual_grpo_action_suffix_advantages(
        rewards=rewards,
        dynamic_gammas=dynamic_gammas,
        dones=dones,
        loss_mask=loss_mask,
        group_size=2,
    )

    assert torch.allclose(result["advantages"], expected_advantages, atol=1e-5)
    assert result["advantages"][0, 1, 1].item() == 0.0
