"""ABot-M0 flow-matching RL head for the XPolicyLab RLinf contract.

Sampling matches official ``AMLFlowMatchingActionHeadRL``: flow-SDE or constant
noise, ``velocity`` / ``x1`` action-head outputs, and a value head on the
concatenated vision-language and proprioceptive features. The Qwen-VL backbone
stays frozen. No rlinf import.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

import torch
from torch import nn


@dataclasses.dataclass
class AbotRLConfig:
    noise_method: str = "flow_sde"
    noise_level: float = 0.5
    # ``x1`` matches AML checkpoints that emit the clean action. ``velocity`` uses the decoder output as-is.
    prediction_type: str = "x1"
    # Denoise steps; None keeps the checkpoint's ``num_inference_timesteps``.
    num_steps: int | None = None
    # Env-facing slice of the action chunk used for log-probs.
    action_chunk: int = 50
    add_value_head: bool = False
    value_hidden_sizes: tuple[int, ...] = (1024, 512, 256)
    joint_logprob: bool = False
    ignore_last: bool = False
    # Fixed Qwen token length so rollout batches stack; pads are masked out of cross-attention.
    qwen_max_length: int = 384

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "AbotRLConfig":
        cfg = dict(cfg or {})
        names = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(cfg) - names)
        if unknown:
            raise ValueError(f"Unknown Abot_M0 rl_head keys: {unknown}")
        if "value_hidden_sizes" in cfg:
            cfg["value_hidden_sizes"] = tuple(int(size) for size in cfg["value_hidden_sizes"])
        config = cls(**cfg)
        if not config.noise_method:
            raise ValueError("noise_method must be 'flow_sde' or another constant-std method.")
        if config.prediction_type not in ("velocity", "x1"):
            raise ValueError(
                f"prediction_type must be 'velocity' or 'x1', got {config.prediction_type!r}."
            )
        return config


class ValueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU()]
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        for module in self.mlp:
            if not isinstance(module, nn.Linear):
                continue
            if module is self.mlp[-1]:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AbotRLPolicy(nn.Module):
    """Frozen Qwen-VL encoder + trainable ABot action head with RL sampling utilities."""

    def __init__(self, base_model: nn.Module, rl_config: AbotRLConfig):
        super().__init__()
        if getattr(base_model, "spatial_model", None) is not None:
            raise NotImplementedError("Abot_M0 RL supports use_vggt=false checkpoints only.")
        self.base_model = base_model
        self.rl_config = rl_config
        action_head = base_model.action_model
        if rl_config.num_steps is not None:
            action_head.num_inference_timesteps = int(rl_config.num_steps)
        self.num_steps = int(action_head.num_inference_timesteps)

        qwen = base_model.qwen_vl_interface
        qwen.to(torch.bfloat16)
        qwen.eval()
        for param in qwen.parameters():
            param.requires_grad = False
        # RL runs without proprio state (include_state=false), so the state encoder never runs.
        if getattr(action_head, "state_encoder", None) is not None:
            action_head.state_encoder.requires_grad_(False)
        # Dropout would make the training-mode recompute differ from the rollout log-prob.
        for module in action_head.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        if rl_config.add_value_head:
            hidden = qwen.model.config.text_config.hidden_size
            # Official head concatenates pooled VL features with pooled state, so the input is 2H.
            self.value_head = ValueHead(hidden * 2, rl_config.value_hidden_sizes)
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.split(".")[-1]

    @property
    def action_head(self) -> nn.Module:
        return self.base_model.action_model

    # FSDP auto-wrap targets. Each unit must hold one dtype and one requires_grad
    # state: frozen bf16 Qwen, trainable fp32 action head, frozen fp32 state encoder.
    # ``future_tokens`` stays in the root unit because its weight is read directly.
    @property
    def _no_split_modules(self) -> list[str]:
        qwen_modules = getattr(self.base_model.qwen_vl_interface.model, "_no_split_modules", None) or []
        return [*qwen_modules, "BasicTransformerBlock", "DiT", "ValueHead"]

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "qwen_vl_interface",
            "action_encoder",
            "action_decoder",
            "position_embedding",
            "state_encoder",
            "value_head",
        ]

    @torch.no_grad()
    def encode(self, batch_images, instructions) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Qwen last hidden states ``[B, L, H]`` and the valid-token mask ``[B, L]``."""
        qwen = self.base_model.qwen_vl_interface
        qwen.eval()
        inputs = qwen.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, max_length=self.rl_config.qwen_max_length
        )
        mask = inputs["attention_mask"].bool()
        if int(mask.sum(dim=1).max()) >= mask.shape[1]:
            raise ValueError(
                f"Qwen inputs fill qwen_max_length={self.rl_config.qwen_max_length}; raise rl_head.qwen_max_length."
            )
        device = next(qwen.parameters()).device
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = qwen(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1].to(torch.bfloat16), mask.to(device)

    @torch.no_grad()
    def sample_actions_rl(self, vl_embs, vl_mask, *, mode: str = "train", compute_values: bool = True) -> dict:
        """Sample one normalized action chunk and record the denoising chain.

        ``mode="train"`` injects noise at one random step (or every step with
        ``joint_logprob``); ``mode="eval"`` is the deterministic AML Euler sampler.
        """
        head = self.action_head
        num_steps = self.num_steps
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(bsize, head.action_horizon, head.action_dim, device=device)
        chains = [x_t]
        log_probs = []
        if self.rl_config.joint_logprob:
            log_probs.append(self._gaussian_logprob(x_t, torch.zeros_like(x_t), torch.ones_like(x_t)))

        if mode == "train":
            if self.rl_config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                high = num_steps - 1
                if self.rl_config.noise_method == "flow_sde" and self.rl_config.ignore_last:
                    high = max(num_steps - 2, 0)
                denoise_inds = torch.full((num_steps,), random.randint(0, high))
        else:
            denoise_inds = torch.full((num_steps,), -1)
        denoise_inds = denoise_inds[None].repeat(bsize, 1).to(device)

        for idx in range(num_steps):
            stochastic = mode == "train" and idx == int(denoise_inds[0, idx])
            x_t_mean, x_t_std = self._step_mean_std(x_t, idx, vl_embs, vl_mask, stochastic)
            x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
            log_probs.append(self._gaussian_logprob(x_t, x_t_mean, x_t_std))
            chains.append(x_t)

        log_probs = torch.stack(log_probs, dim=1)[:, :, : self.rl_config.action_chunk]
        if self.rl_config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[torch.arange(bsize, device=device), denoise_inds[:, 0]]
        values = self._value(vl_embs, vl_mask) if compute_values else torch.zeros(bsize, device=device)
        return {
            "actions": x_t,
            "chains": torch.stack(chains, dim=1),
            "denoise_inds": denoise_inds,
            "prev_logprobs": log_probs,
            "prev_values": values[:, None],
        }

    def get_log_prob_value(self, vl_embs, vl_mask, chains, denoise_inds, *, compute_values=False) -> dict:
        """Recompute log-probs / values of a recorded chain (actor side, with grad)."""
        bsize = chains.shape[0]
        batch_indices = torch.arange(bsize, device=chains.device)
        chain_log_probs = []
        if self.rl_config.joint_logprob:
            num_steps = self.num_steps
            chain_log_probs.append(
                self._gaussian_logprob(chains[:, 0], torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0]))
            )
        else:
            num_steps = 1
        for step in range(num_steps):
            denoise_ind = denoise_inds[:, step]
            x_t_mean, x_t_std = self._step_mean_std(
                chains[batch_indices, denoise_ind], denoise_ind, vl_embs, vl_mask, stochastic=True
            )
            chain_log_probs.append(self._gaussian_logprob(chains[batch_indices, denoise_ind + 1], x_t_mean, x_t_std))

        log_probs = torch.stack(chain_log_probs, dim=1)[:, :, : self.rl_config.action_chunk]
        values = self._value(vl_embs, vl_mask) if compute_values else torch.zeros(bsize, device=chains.device)
        return {
            "logprobs": log_probs.mean(dim=1),
            "values": values,
            "entropy": torch.zeros((bsize, 1), device=chains.device),
        }

    def _step_mean_std(self, x_t, idx, vl_embs, vl_mask, stochastic: bool):
        """Mean / std of the next AML Euler state; ``idx`` is an int or a per-sample tensor."""
        head = self.action_head
        bsize = x_t.shape[0]
        device = x_t.device
        num_steps = self.num_steps
        if isinstance(idx, int):
            idx = torch.full((bsize,), idx, device=device, dtype=torch.long)
        idx = idx.to(device=device, dtype=torch.long)
        model_dtype = next(head.model.parameters()).dtype
        x_t = x_t.to(torch.float32)

        t_cont = idx.to(torch.float32) / num_steps
        timesteps = (t_cont * head.num_timestep_buckets).long().clamp(0, head.num_timestep_buckets - 1)
        action_features = head.action_encoder(x_t.to(model_dtype), timesteps)
        if head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(bsize, -1, -1)
        sa_embs = torch.cat((future_tokens, action_features), dim=1).to(model_dtype)
        model_output = head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs.to(model_dtype),
            timestep=timesteps,
            encoder_attention_mask=vl_mask,
        )
        decoded = head.action_decoder(model_output)[:, -head.action_horizon :].to(torch.float32)

        t = t_cont[:, None, None]
        dt = 1.0 / num_steps
        if self.rl_config.prediction_type == "velocity":
            velocity = decoded
        else:
            velocity = (decoded - x_t) / (1.0 - t).clamp_min(head.t_eps)
        x0_pred = x_t - velocity * t
        x1_pred = x_t + velocity * (1.0 - t)
        if not stochastic:
            return x0_pred * (1.0 - (t + dt)) + x1_pred * (t + dt), torch.zeros_like(x_t)
        if self.rl_config.noise_method == "flow_sde":
            grid = torch.linspace(0, 1, num_steps + 1, device=device)
            sigmas = self.rl_config.noise_level * torch.sqrt((1 - grid) / torch.where(grid == 0, grid[1], grid))[:-1]
            sigma = sigmas[idx][:, None, None]
            x0_weight = 1.0 - (t + dt) - sigma**2 * dt / (2 * (1.0 - t).clamp_min(1e-8))
            x1_weight = t + dt
            return x0_pred * x0_weight + x1_pred * x1_weight, math.sqrt(dt) * sigma.expand_as(x_t)
        return x_t + dt * velocity, torch.full_like(x_t, self.rl_config.noise_level)

    def _value(self, vl_embs, vl_mask, state_features=None) -> torch.Tensor:
        if not self.rl_config.add_value_head:
            return torch.zeros(vl_embs.shape[0], device=vl_embs.device)
        del vl_mask
        pooled_vl = vl_embs.to(torch.float32).mean(dim=1)
        if state_features is None:
            pooled_state = torch.zeros_like(pooled_vl)
        else:
            flat = state_features.to(device=pooled_vl.device, dtype=pooled_vl.dtype).reshape(pooled_vl.shape[0], -1)
            pooled_state = torch.nn.functional.adaptive_avg_pool1d(flat.unsqueeze(1), pooled_vl.shape[-1]).squeeze(1)
        value_in = torch.cat((pooled_vl, pooled_state), dim=1)
        return self.value_head(value_in.to(self.value_head.mlp[0].weight.dtype))[:, 0].to(torch.float32)

    @staticmethod
    def _gaussian_logprob(sample, mu, sigma) -> torch.Tensor:
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        log_prob = (
            -torch.log(sigma_safe)
            - 0.5 * math.log(2 * math.pi)
            - 0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        )
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)
