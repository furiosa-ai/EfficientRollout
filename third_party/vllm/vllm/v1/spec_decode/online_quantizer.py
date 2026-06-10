"""Online quantizer for self-speculative decoding with periodic requantization.

Supports 3 tiers:
- Tier 0 (RTN): Round-to-nearest, no calibration, sub-second
- Tier 1 (fixed_beta): Activation-aware scaling with fixed β=0.5, sub-second
- Tier 2 (replay_awq): Faithful AWQ with stored rollout data, runs every K steps

The quantizer manages the lifecycle of quantized drafter weights,
coordinating with the veRL training loop's weight sync cycle.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OnlineQuantizer:
    """Manages online W4 quantization with multi-tier support.

    Called after each weight update to requantize the drafter model
    from the target model's FP16 weights.
    """

    def __init__(
        self,
        method: str = "rtn",
        group_size: int = 128,
        quant_interval: int = 1,
        replay_interval: int = 10,
    ):
        self.method = method  # "rtn" | "fixed_beta" | "replay_awq"
        self.group_size = group_size
        self.quant_interval = quant_interval
        self.replay_interval = replay_interval
        self.step_counter = 0

        # Tier 1: activation statistics from previous rollout
        self.activation_stats: dict[str, torch.Tensor] = {}

        # Tier 2: stored rollout tokens for calibration replay
        self.calibration_tokens: Optional[torch.Tensor] = None

        logger.info(
            "OnlineQuantizer initialized: method=%s, group_size=%d, "
            "quant_interval=%d",
            method, group_size, quant_interval,
        )

    def should_requantize(self) -> bool:
        """Check if requantization is needed this step."""
        return self.step_counter % self.quant_interval == 0

    def _get_scales(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get per-channel scaling factors for activation-aware quantization.

        For RTN (Tier 0): returns None (no scaling)
        For fixed_beta (Tier 1): returns stats.pow(0.5) from previous rollout
        For replay_awq (Tier 2): returns None (AWQ uses different pipeline)
        """
        if self.method == "rtn":
            return None

        if self.method == "fixed_beta":
            if layer_name in self.activation_stats:
                act_stats = self.activation_stats[layer_name]
                # Fixed β=0.5 (AWQ paper: most layers optimal near 0.5)
                return act_stats.pow(0.5)
            return None  # Fallback to RTN if no stats available

        # replay_awq: calibration is done separately
        return None

    def update_activation_stats(self, stats: dict[str, torch.Tensor]):
        """Update activation statistics from rollout forward pass.

        Called after each rollout to provide calibration data for next step.
        """
        self.activation_stats = stats
        logger.debug(
            "Updated activation stats for %d layers", len(stats)
        )

    def store_rollout_tokens(self, tokens: torch.Tensor):
        """Store rollout tokens for Tier 2 Replay-AWQ calibration."""
        self.calibration_tokens = tokens

    def should_replay_awq(self) -> bool:
        """Check if full AWQ recalibration should run (Tier 2, every K steps)."""
        return (
            self.method == "replay_awq"
            and self.step_counter % self.replay_interval == 0
            and self.calibration_tokens is not None
        )


class ActivationStatsCollector:
    """Collect per-channel activation magnitudes during rollout forward.

    Registers forward hooks on AWQ-relevant linear layer inputs.
    Hook points follow autoawq's Qwen2 scaling pairs:
    - self_attn.q_proj input (covers QKV projection)
    - self_attn.o_proj input
    - mlp.gate_proj input (covers gate/up projection)
    - mlp.down_proj input

    Usage:
        collector = ActivationStatsCollector(model)
        # ... run rollout forward passes ...
        stats = collector.collect_and_clear()
        quantizer.update_activation_stats(stats)
        collector.remove_hooks()  # when done
    """

    # AWQ scaling pair hook points for Qwen2/LLaMA architectures
    HOOK_SUFFIXES = [
        "self_attn.q_proj",   # input to QKV
        "self_attn.o_proj",   # input to O
        "mlp.gate_proj",      # input to gate/up
        "mlp.down_proj",      # input to down
    ]

    def __init__(self, model: nn.Module):
        self.stats: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._count: dict[str, int] = {}  # running average counter
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        """Register forward hooks on AWQ-relevant linear inputs."""
        for name, module in model.named_modules():
            if any(name.endswith(suffix) for suffix in self.HOOK_SUFFIXES):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

        logger.info(
            "ActivationStatsCollector: registered %d hooks", len(self._hooks)
        )

    def _make_hook(self, name: str):
        """Create a hook that collects per-channel activation magnitude."""
        def hook_fn(module, input, output):
            x = input[0]
            if isinstance(x, tuple):
                x = x[0]
            x = x.detach()
            # Per-channel mean absolute value: [hidden_dim]
            channel_mag = x.float().abs().mean(dim=tuple(range(x.dim() - 1)))

            if name in self.stats:
                # Running average to avoid memory growth
                count = self._count[name]
                self.stats[name] = (
                    self.stats[name] * count + channel_mag
                ) / (count + 1)
                self._count[name] = count + 1
            else:
                self.stats[name] = channel_mag
                self._count[name] = 1
        return hook_fn

    def collect_and_clear(self) -> dict[str, torch.Tensor]:
        """Return collected stats and clear internal state."""
        stats = dict(self.stats)
        self.stats.clear()
        self._count.clear()
        return stats

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("ActivationStatsCollector: removed all hooks")
