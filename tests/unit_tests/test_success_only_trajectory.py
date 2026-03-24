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

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.utils.nested_dict_process import infer_batch_size


def _build_success_trajectory():
    traj_len = 4
    batch_size = 2

    actions = torch.arange(traj_len * batch_size, dtype=torch.float32).view(
        traj_len, batch_size, 1
    )
    rewards = torch.ones(traj_len, batch_size, 1, dtype=torch.float32)
    prev_logprobs = torch.zeros(traj_len, batch_size, 1, dtype=torch.float32)
    versions = torch.zeros(traj_len, batch_size, 1, dtype=torch.float32)
    forward_inputs = {
        "model_action": torch.arange(
            traj_len * batch_size * 2, dtype=torch.float32
        ).view(traj_len, batch_size, 2)
    }

    # One extra bootstrap row matches the common rollout storage layout.
    terminations = torch.tensor(
        [
            [[False], [False]],
            [[False], [False]],
            [[True], [False]],
            [[False], [False]],
            [[False], [True]],
        ],
        dtype=torch.bool,
    )
    dones = terminations.clone()
    truncations = torch.zeros_like(terminations)

    return Trajectory(
        max_episode_length=traj_len,
        model_weights_id="test",
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        dones=dones,
        prev_logprobs=prev_logprobs,
        versions=versions,
        forward_inputs=forward_inputs,
    )


def test_extract_success_traj_truncates_after_first_success():
    traj = _build_success_trajectory()

    extracted = traj.extract_success_traj(truncate_after_first_success=True)

    assert extracted is not None
    assert len(extracted) == 2

    first_traj, second_traj = extracted
    assert first_traj.actions.shape == (2, 1, 1)
    assert second_traj.actions.shape == (4, 1, 1)

    assert torch.equal(first_traj.actions[:, 0, 0], torch.tensor([0.0, 2.0]))
    assert torch.equal(second_traj.actions[:, 0, 0], torch.tensor([1.0, 3.0, 5.0, 7.0]))

    assert first_traj.terminations.shape == (2, 1, 1)
    assert bool(first_traj.terminations[-1, 0, 0]) is True
    assert bool(second_traj.terminations[-1, 0, 0]) is True


def test_extract_success_traj_keeps_full_prefix_when_not_truncating():
    traj = _build_success_trajectory()

    extracted = traj.extract_success_traj(truncate_after_first_success=False)

    assert extracted is not None
    assert len(extracted) == 2
    assert all(item.actions.shape == (4, 1, 1) for item in extracted)

    first_traj = extracted[0]
    assert first_traj.forward_inputs["model_action"].shape == (4, 1, 2)
    assert torch.equal(
        first_traj.forward_inputs["model_action"][:, 0, 0],
        torch.tensor([0.0, 4.0, 8.0, 12.0]),
    )


def test_infer_batch_size_from_nested_dict():
    batch = {
        "forward_inputs": {
            "observation/main_images": torch.zeros(3, 2, 2),
            "action": torch.zeros(3, 10),
        },
        "curr_obs": {"states": torch.zeros(3, 7)},
    }

    assert infer_batch_size(batch) == 3
