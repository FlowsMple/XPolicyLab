"""ABot sampling and update regression without Qwen weights."""

import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from XPolicyLab.policy.Abot_M0.rl import AbotRLConfig, AbotRLPolicy


class ActionEncoder(nn.Linear):
    def forward(self, actions, timesteps):
        return super().forward(actions)


class Conditioner(nn.Linear):
    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        return super().forward(hidden_states + encoder_hidden_states.mean(1)[:, None])


def small_abot(value, joint):
    base = nn.Module()
    base.qwen_vl_interface = nn.Module()
    base.qwen_vl_interface.model = nn.Module()
    base.qwen_vl_interface.model.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=4)
    )
    head = base.action_model = nn.Module()
    head.num_inference_timesteps = 3
    head.action_horizon = 3
    head.action_dim = 4
    head.num_timestep_buckets = 100
    head.t_eps = 1e-5
    head.config = SimpleNamespace(add_pos_embed=False)
    head.action_encoder = ActionEncoder(4, 4)
    head.future_tokens = nn.Embedding(1, 4)
    head.model = Conditioner(4, 4)
    head.action_decoder = nn.Linear(4, 4)
    return AbotRLPolicy(
        base, AbotRLConfig(action_chunk=2, add_value_head=value, joint_logprob=joint)
    )


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_abot_real_likelihood_gradient_and_update(value, joint):
    torch.manual_seed(42)
    random.seed(42)
    model = small_abot(value, joint)
    features, mask = torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool)
    sampled = model.sample_actions_rl(features, mask, compute_values=value)
    result = model.get_log_prob_value(
        features, mask, sampled["chains"], sampled["denoise_inds"], compute_values=value
    )
    torch.testing.assert_close(result["logprobs"], sampled["prev_logprobs"])
    torch.testing.assert_close(result["values"], sampled["prev_values"][:, 0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    (-result["logprobs"].mean() + result["values"].square().mean()).backward()
    assert model.action_head.action_decoder.weight.grad.abs().sum() > 0
    optimizer.step()
    updated = model.get_log_prob_value(
        features, mask, sampled["chains"], sampled["denoise_inds"]
    )
    assert not torch.allclose(updated["logprobs"], sampled["prev_logprobs"])
