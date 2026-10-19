"""Pi0.5 flow-matching RL head for the XPolicyLab RLinf contract.

Sampling matches official ``Pi0RL``: flow-ODE / flow-SDE / flow-CPS / learned
``flow_noise``, a value head on the action expert or the VLM prefix, and the
tensors NFT needs to recompute velocity. Pure torch, no rlinf import.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

NOISE_METHODS = ("flow_ode", "flow_sde", "flow_cps", "flow_noise")
VALUE_VLM_MODES = ("mean_token", "first_token", "last_token")
_VLM_VALUE_CONFIGS = ("pi05_maniskill", "pi05_libero", "pi05_droid_polaris")


@dataclasses.dataclass
class Pi05RLConfig:
    noise_method: str = "flow_sde"
    noise_level: float = 0.5
    noise_logvar_range: tuple[float, float] = (0.08, 0.16)
    num_steps: int = 10
    # Env-facing slice of the model output used for log-probs: [:action_chunk, :action_env_dim].
    action_chunk: int = 50
    action_env_dim: int = 14
    add_value_head: bool = False
    value_hidden_sizes: tuple[int, ...] | None = None
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"
    detach_critic_input: bool = False
    chunk_critic_input: bool = False
    joint_logprob: bool = False
    ignore_last: bool = False
    safe_get_logprob: bool = False
    train_expert_only: bool = False
    is_nft: bool = False
    config_name: str = ""

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "Pi05RLConfig":
        cfg = dict(cfg or {})
        names = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(cfg) - names)
        if unknown:
            raise ValueError(f"Unknown Pi_05 rl_head keys: {unknown}")
        if "value_hidden_sizes" in cfg and cfg["value_hidden_sizes"] is not None:
            cfg["value_hidden_sizes"] = tuple(int(size) for size in cfg["value_hidden_sizes"])
        if "noise_logvar_range" in cfg:
            cfg["noise_logvar_range"] = tuple(float(value) for value in cfg["noise_logvar_range"])
        config = cls(**cfg)
        if config.noise_method not in NOISE_METHODS:
            raise ValueError(f"noise_method must be one of {NOISE_METHODS}, got {config.noise_method!r}.")
        if config.value_vlm_mode not in VALUE_VLM_MODES:
            raise ValueError(f"value_vlm_mode must be one of {VALUE_VLM_MODES}, got {config.value_vlm_mode!r}.")
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


class _ExploreNoiseNet(nn.Module):
    """Learned per-dimension exploration std. Same map as official ``ExploreNoiseNet``."""

    def __init__(self, in_dim: int, out_dim: int, noise_logvar_range: tuple[float, float]):
        super().__init__()
        self.mlp_logvar = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
        )
        logvar_min = math.log(noise_logvar_range[0] ** 2)
        logvar_max = math.log(noise_logvar_range[1] ** 2)
        self.register_buffer("logvar_min", torch.tensor([logvar_min], dtype=torch.float32))
        self.register_buffer("logvar_max", torch.tensor([logvar_max], dtype=torch.float32))

    def forward(self, noise_feature: torch.Tensor) -> torch.Tensor:
        noise_logvar = torch.tanh(self.mlp_logvar(noise_feature.to(torch.float32)))
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2.0
        return torch.exp(0.5 * noise_logvar)


class Pi05RLPolicy(PI0Pytorch):
    """``PI0Pytorch`` plus flow-SDE sampling, log-prob recompute and a value head."""

    # FSDP auto-wrap targets. RMSNorm / patch embeddings stay fp32 inside a bf16
    # backbone, so they must be separate flat params.
    _NO_SPLIT_MODULES = ("GemmaMLP", "SiglipVisionEmbeddings", "GemmaRMSNorm", "GemmaRotaryEmbedding")
    _NO_SPLIT_MODULES_EXPERT_ONLY = (
        "GemmaDecoderLayer",
        "SiglipVisionEmbeddings",
        "GemmaRMSNorm",
        "GemmaRotaryEmbedding",
    )
    _NO_SPLIT_NAMES = (
        "action_in_proj",
        "action_out_proj",
        "lm_head",
        "time_mlp_in",
        "time_mlp_out",
        "value_head",
        "noise_head",
    )

    def __init__(self, model_config, rl_config: Pi05RLConfig):
        if not model_config.pi05:
            raise ValueError("Pi05RLPolicy requires a pi05 model config.")
        # Rollout calls the denoise loop step by step; compiling ``sample_actions`` is not used.
        super().__init__(dataclasses.replace(model_config, pytorch_compile_mode=None))
        self.rl_config = rl_config
        if rl_config.add_value_head:
            if rl_config.value_after_vlm:
                width = self.paligemma_with_expert.paligemma.language_model.config.hidden_size
            else:
                width = self.action_in_proj.out_features
            hidden = rl_config.value_hidden_sizes
            if hidden is None:
                hidden = (1024, 512, 256) if rl_config.value_after_vlm else (512, 256, 128)
                if any(name in rl_config.config_name for name in _VLM_VALUE_CONFIGS):
                    hidden = (1024, 512, 256)
            self.value_head = ValueHead(width, hidden)
        if rl_config.noise_method == "flow_noise":
            self.noise_head = _ExploreNoiseNet(
                self.action_in_proj.out_features,
                self.config.action_dim,
                rl_config.noise_logvar_range,
            )
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.split(".")[-1]

    @property
    def _no_split_modules(self) -> list[str]:
        if self.rl_config.train_expert_only:
            return list(self._NO_SPLIT_MODULES_EXPERT_ONLY)
        return list(self._NO_SPLIT_MODULES)

    @property
    def _no_split_names(self) -> list[str]:
        return list(self._NO_SPLIT_NAMES)

    def freeze_vlm(self) -> None:
        self.paligemma_with_expert.paligemma.eval()
        for param in self.paligemma_with_expert.paligemma.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def sample_actions_rl(self, observation, *, mode: str = "train", compute_values: bool = True) -> dict:
        """Sample one action chunk and record the denoising chain for PPO / GRPO.

        ``mode="train"`` injects noise at one random denoise step (or every step with
        ``joint_logprob``); ``mode="eval"`` is the deterministic flow ODE.
        """
        rl_config = self.rl_config
        num_steps = rl_config.num_steps
        state, prefix_output, prefix_pad_masks, past_key_values = self._encode_prefix(observation)
        bsize = state.shape[0]
        device = state.device

        x_t = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)
        chains = [x_t]
        log_probs = []
        values = []
        if rl_config.joint_logprob:
            log_probs.append(self._gaussian_logprob(x_t, torch.zeros_like(x_t), torch.ones_like(x_t)))
        vlm_value = None
        if compute_values and rl_config.add_value_head and rl_config.value_after_vlm:
            vlm_value = self._value_from_prefix(prefix_output, prefix_pad_masks)

        if mode == "train" and (rl_config.joint_logprob or rl_config.is_nft):
            denoise_inds = torch.arange(num_steps)
        elif mode == "train":
            high = num_steps - 2 if rl_config.ignore_last else num_steps - 1
            denoise_inds = torch.tensor([random.randint(0, max(high, 0))] * num_steps)
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)
        nft_state = self._init_nft_state(x_t, num_steps) if rl_config.is_nft else None

        for idx in range(num_steps):
            method = rl_config.noise_method if idx == int(denoise_inds[0, idx]) else "flow_ode"
            x_t_prev = x_t
            x_t_mean, x_t_std, value_t, velocity = self._step_mean_std_value(
                x_t, idx, state, prefix_pad_masks, past_key_values, method, compute_values
            )
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            self._update_nft_state(nft_state, idx, x_t_prev, velocity, x_t, method)
            log_probs.append(self._gaussian_logprob(x_t, x_t_mean, x_t_std))
            if value_t is not None:
                values.append(value_t)
            chains.append(x_t)

        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : rl_config.action_chunk, : rl_config.action_env_dim
        ]
        if rl_config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[torch.arange(bsize, device=device), denoise_inds[:, 0]]
        if vlm_value is not None:
            prev_values = vlm_value[:, None]
        elif values:
            prev_values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        else:
            prev_values = torch.zeros((bsize, 1), device=device)
        result = {
            "actions": x_t,
            "chains": chains,
            "denoise_inds": denoise_inds,
            "prev_logprobs": log_probs,
            "prev_values": prev_values,
        }
        if nft_state is not None:
            result.update(nft_state)
            result["nft_x0"] = x_t.detach()
        return result

    def get_log_prob_value(
        self,
        observation,
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        *,
        compute_values: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Recompute log-probs / values of a recorded chain (actor side, with grad)."""
        state, prefix_output, prefix_pad_masks, past_key_values = self._encode_prefix(observation)
        bsize = state.shape[0]
        batch_indices = torch.arange(bsize, device=chains.device)
        denoise_inds = denoise_inds.to(chains.device)

        rl_config = self.rl_config
        chain_log_probs = []
        chain_values = []
        chain_entropy = []
        learned_noise = rl_config.noise_method == "flow_noise"
        if rl_config.joint_logprob:
            num_steps = rl_config.num_steps
            chain_log_probs.append(
                self._gaussian_logprob(chains[:, 0], torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0]))
            )
            if learned_noise:
                chain_entropy.append(self._gaussian_entropy(torch.ones_like(chains[:, 0])))
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            x_t_mean, x_t_std, value_t, _velocity = self._step_mean_std_value(
                chains[batch_indices, denoise_ind],
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                rl_config.noise_method,
                compute_values,
            )
            chain_log_probs.append(self._gaussian_logprob(chains[batch_indices, denoise_ind + 1], x_t_mean, x_t_std))
            if value_t is not None:
                chain_values.append(value_t)
            if learned_noise:
                chain_entropy.append(self._gaussian_entropy(x_t_std))

        log_probs = torch.stack(chain_log_probs, dim=1)[
            :, :, : rl_config.action_chunk, : rl_config.action_env_dim
        ]
        if compute_values and rl_config.add_value_head and rl_config.value_after_vlm:
            values = self._value_from_prefix(prefix_output, prefix_pad_masks)
        elif chain_values:
            values = torch.stack(chain_values, dim=1).mean(dim=1)
        else:
            values = torch.zeros(bsize, device=chains.device)
        if learned_noise:
            entropy = torch.stack(chain_entropy, dim=1)[
                :, :, : rl_config.action_chunk, : rl_config.action_env_dim
            ]
            entropy = entropy.mean(dim=(1, 2, 3))[:, None]
        else:
            entropy = torch.zeros((bsize, 1), device=chains.device)
        return {
            "logprobs": log_probs.mean(dim=1),
            "values": values,
            "entropy": entropy,
        }

    def nft_forward(self, observation, nft_inputs: dict[str, torch.Tensor], compute_values: bool = False) -> dict:
        """Velocity at an explicit ``(x_t, timesteps)`` pair for the NFT loss."""
        x_t = nft_inputs["x_t"]
        timesteps = nft_inputs["timesteps"]
        if timesteps.dim() > 1:
            timesteps = timesteps.reshape(timesteps.shape[0], -1)[:, 0]
        state, prefix_output, prefix_pad_masks, past_key_values = self._encode_prefix(observation)
        suffix_out = self._suffix_out(state, prefix_pad_masks, past_key_values, x_t, timesteps)
        velocity = self.action_out_proj(suffix_out.to(self.action_out_proj.weight.dtype))
        result = {
            "v_theta": velocity[:, : self.rl_config.action_chunk],
            "x_t": x_t,
            "timesteps": timesteps,
        }
        if compute_values and self.rl_config.add_value_head:
            if self.rl_config.value_after_vlm:
                result["values"] = self._value_from_prefix(prefix_output, prefix_pad_masks)[:, None]
            else:
                result["values"] = self._value_from_suffix(suffix_out)[:, None]
        return result

    def _encode_prefix(self, observation):
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return state, prefix_output, prefix_pad_masks, past_key_values

    def _suffix_out(self, state, prefix_pad_masks, past_key_values, x_t, timestep) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)

    def _step_mean_std_value(self, x_t, idx, state, prefix_pad_masks, past_key_values, method, compute_values):
        """Mean / std of ``x_{t-dt}`` given ``x_t`` for one Euler step, plus the value estimate.

        ``idx`` is an int during rollout and a per-sample tensor during recompute.
        """
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.full((bsize,), idx, device=device)
        num_steps = self.rl_config.num_steps
        timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)
        timesteps = torch.cat([timesteps, torch.zeros(1, device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self._suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_input)
        v_t = self.action_out_proj(suffix_out.to(self.action_out_proj.weight.dtype)).to(torch.float32)
        suffix_value = (
            compute_values and self.rl_config.add_value_head and not self.rl_config.value_after_vlm
        )
        value_t = self._value_from_suffix(suffix_out) if suffix_value else None

        x_t = x_t.to(torch.float32)
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        noise_level = torch.tensor(self.rl_config.noise_level, device=device)
        if method in ("flow_ode", "flow_noise"):
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input) if method == "flow_ode" else self.noise_head(suffix_out)
        elif method == "flow_sde":
            denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigmas = noise_level * torch.sqrt(timesteps / (1 - denom_timesteps))[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        elif method == "flow_cps":
            x0_weight = 1 - (t_input - delta)
            x1_weight = (t_input - delta) * torch.cos(math.pi * noise_level / 2)
            x_t_std = (t_input - delta) * torch.sin(math.pi * noise_level / 2)
        else:
            raise ValueError(f"Invalid noise method: {method}")
        return x0_pred * x0_weight + x1_pred * x1_weight, x_t_std, value_t, v_t

    def _value_from_suffix(self, suffix_out: torch.Tensor) -> torch.Tensor:
        if self.rl_config.chunk_critic_input:
            suffix_out = suffix_out[:, : self.rl_config.action_chunk]
        value_in = suffix_out.mean(dim=1)
        if self.rl_config.detach_critic_input:
            value_in = value_in.detach()
        return self.value_head(value_in.to(self.value_head.mlp[0].weight.dtype))[:, 0].to(torch.float32)

    def _value_from_prefix(self, prefix_out: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        mask = prefix_mask.to(prefix_out.dtype)
        mode = self.rl_config.value_vlm_mode
        if mode == "mean_token":
            weight = mask.unsqueeze(-1)
            pooled = (prefix_out * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        elif mode == "first_token":
            index = mask.to(torch.long).argmax(dim=1)
            pooled = prefix_out[torch.arange(prefix_out.shape[0], device=prefix_out.device), index]
        else:
            reversed_mask = torch.flip(mask, dims=[1])
            last_from_end = reversed_mask.to(torch.long).argmax(dim=1)
            index = mask.shape[1] - 1 - last_from_end
            pooled = prefix_out[torch.arange(prefix_out.shape[0], device=prefix_out.device), index]
        return self.value_head(pooled.to(self.value_head.mlp[0].weight.dtype))[:, 0].to(torch.float32)

    def _init_nft_state(self, x_t: torch.Tensor, num_steps: int) -> dict[str, torch.Tensor]:
        return {
            "nft_step_index": torch.randint(0, num_steps, (x_t.shape[0],), device=x_t.device),
            "nft_xcur": torch.zeros_like(x_t),
            "nft_v": torch.zeros_like(x_t),
            "nft_xnext": torch.zeros_like(x_t),
            "nft_noise_level": torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype),
        }

    def _update_nft_state(self, nft_state, idx, x_t_prev, velocity, x_t, method) -> None:
        if nft_state is None:
            return
        mask = nft_state["nft_step_index"] == idx
        if not mask.any():
            return
        mask_bc = mask[:, None, None]
        nft_state["nft_xcur"] = torch.where(mask_bc, x_t_prev.detach(), nft_state["nft_xcur"])
        nft_state["nft_v"] = torch.where(mask_bc, velocity.detach(), nft_state["nft_v"])
        nft_state["nft_xnext"] = torch.where(mask_bc, x_t.detach(), nft_state["nft_xnext"])
        level = 0.0 if method == "flow_ode" else float(self.rl_config.noise_level)
        nft_state["nft_noise_level"] = torch.where(
            mask, torch.full_like(nft_state["nft_noise_level"], level), nft_state["nft_noise_level"]
        )

    def _gaussian_logprob(self, sample, mu, sigma) -> torch.Tensor:
        if self.rl_config.safe_get_logprob:
            return -torch.pow(sample - mu, 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        log_prob = (
            -torch.log(sigma_safe)
            - 0.5 * math.log(2 * math.pi)
            - 0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        )
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    @staticmethod
    def _gaussian_entropy(sigma: torch.Tensor) -> torch.Tensor:
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * sigma_safe**2)
        return torch.where(mask, torch.zeros_like(entropy), entropy)


def load_pi05_rl_policy(model_config, weight_path: str, rl_config: Pi05RLConfig) -> Pi05RLPolicy:
    """Build ``Pi05RLPolicy`` and load an upstream openpi PyTorch ``model.safetensors``."""
    policy = Pi05RLPolicy(model_config, rl_config)
    missing, unexpected = safetensors.torch.load_model(policy, weight_path, strict=False)
    # embed_tokens is tied to the PaliGemma lm_head and not stored separately.
    missing = [
        key
        for key in missing
        if not key.startswith(("value_head.", "noise_head.")) and not key.endswith("embed_tokens.weight")
    ]
    if missing or unexpected:
        raise RuntimeError(f"Pi_05 checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    policy.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if rl_config.train_expert_only:
        policy.freeze_vlm()
    return policy
