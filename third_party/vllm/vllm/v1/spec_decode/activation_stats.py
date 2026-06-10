"""Activation statistics collector for AWQ-aware requantization.

Phase 0: Collects per-channel mean(|X|) on linear layer inputs via forward
pre-hooks. Stats are accumulated with Welford-style running average (no memory
growth). Collected stats live on GPU during generation, then moved to CPU via
collect_and_clear() before requantization.

Hook targets (matching AWQ paper's salient channel detection):
- self_attn.q_proj  (QKV input)
- self_attn.o_proj  (attention output projection input)
- mlp.gate_proj     (MLP gate/up input)
- mlp.down_proj     (MLP down projection input)

Usage:
    collector = ActivationStatsCollector(model, device, calibration_prompt_num=1)
    collector.enabled = True
    # ... run forward (hooks accumulate stats) ...
    collector.mark_prompt_boundary()  # after each micro-batch
    stats, samples = collector.collect_and_clear()
    # stats: dict[str, Tensor] — per-channel mean(|X|) on CPU
    # samples: dict[str, Tensor] — [num_tokens, in_features] on CPU
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Linear sublayer suffixes to hook (input activations for AWQ scaling).
_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
)


class ActivationStatsCollector:
    """Collect per-channel activation magnitudes for AWQ scale computation.

    Registers forward pre-hooks on target linear layers to accumulate
    per-channel mean(|X|) using a Welford-style running average. This avoids
    storing any activation tensors — only a single [in_features] vector per
    layer is maintained on GPU.

    Activation samples are collected from calibration_prompt_num prompts for
    block output MSE search. Default 1 prompt (~512 tokens). Set to 0 to
    collect only running mean (no samples).

    Thread safety: hooks are called synchronously within the forward pass
    (single writer pattern), so no locking is needed.

    Args:
        model: The model to instrument (drafter or target).
        device: GPU device for stats accumulation.
        calibration_prompt_num: Number of unique prompts to collect samples
            from. Default 1. Set to 0 to disable sample collection.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        calibration_prompt_num: int = 1,
    ) -> None:
        self._device = device
        self._enabled = False
        self._calibration_prompt_num = calibration_prompt_num

        # Running stats: per-channel mean(|X|) accumulated via Welford update.
        # Key: fully-qualified module name (e.g., "model.layers.0.self_attn.q_proj")
        # Value: [in_features] tensor on GPU
        self._running_mean: dict[str, torch.Tensor] = {}

        # Token count per layer (for Welford running average).
        self._token_counts: dict[str, int] = {}

        # Activation samples: collected from up to calibration_prompt_num prompts.
        # Key: module name, Value: list of [num_tokens, in_features] tensors on CPU
        self._activation_samples: dict[str, list[torch.Tensor]] = {}

        # Track how many sample batches have been collected (acts as prompt counter).
        self._sample_batches_collected: int = 0

        # Total tokens seen across all collect cycles (diagnostic counter).
        self._total_tokens_seen: int = 0

        # Hook handles for cleanup.
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

        # Register hooks on target layers.
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward pre-hooks on target linear layers.

        Matches module names ending with any of _TARGET_SUFFIXES.
        """
        registered = 0
        for name, module in model.named_modules():
            if not any(name.endswith(suffix) for suffix in _TARGET_SUFFIXES):
                continue
            if not isinstance(module, nn.Module):
                continue
            # Verify it has weight (i.e., is a linear-like layer).
            has_weight = (
                hasattr(module, 'weight')
                or hasattr(module, 'qweight')  # AWQ quantized layers
            )
            if not has_weight:
                continue

            handle = module.register_forward_pre_hook(
                self._make_hook(name)
            )
            self._hook_handles.append(handle)
            registered += 1

        logger.info(
            "ActivationStatsCollector: registered %d hooks on %d target suffixes",
            registered,
            len(_TARGET_SUFFIXES),
        )

    def _make_hook(self, layer_name: str):
        """Create a forward pre-hook closure for a specific layer.

        The hook computes per-channel mean(|X|) and updates the running
        average using Welford's online algorithm:
            new_mean = old_mean + (batch_mean - old_mean) / n

        This is numerically stable and requires O(in_features) memory
        regardless of how many tokens are processed.
        """

        def hook_fn(module: nn.Module, args: tuple) -> None:
            if not self._enabled:
                return

            # args[0] is the input tensor: [num_tokens, in_features]
            x = args[0]
            if isinstance(x, tuple):
                x = x[0]

            # Per-channel mean of absolute values across token dimension.
            # x shape: [num_tokens, in_features] or [batch, seq, in_features]
            if x.dim() == 3:
                # [batch, seq, features] → merge batch and seq
                x = x.reshape(-1, x.shape[-1])

            num_tokens = x.shape[0]
            # mean(|X|) across tokens → [in_features]
            batch_mean = x.abs().mean(dim=0).float()

            # Store activation samples: 1 prompt ≈ first 512 tokens from
            # the micro-batch (which contains many packed sequences).
            # calibration_prompt_num controls how many micro-batches contribute.
            # Upper bound 512 tokens per sample to prevent CPU OOM
            # (full micro-batch ~24k tokens → ~35GB vs 512 tokens → ~0.9GB).
            _MAX_SAMPLE_TOKENS = 512
            if (self._calibration_prompt_num > 0
                    and self._sample_batches_collected < self._calibration_prompt_num):
                sample = x[:_MAX_SAMPLE_TOKENS].detach().cpu().float()
                if layer_name not in self._activation_samples:
                    self._activation_samples[layer_name] = [sample]
                else:
                    self._activation_samples[layer_name].append(sample)

            # Welford running average update.
            if layer_name not in self._running_mean:
                self._running_mean[layer_name] = batch_mean.to(self._device)
                self._token_counts[layer_name] = num_tokens
            else:
                old_count = self._token_counts[layer_name]
                new_count = old_count + num_tokens
                # Weighted merge: old contributes old_count, new contributes num_tokens
                old_mean = self._running_mean[layer_name]
                self._running_mean[layer_name] = (
                    old_mean * (old_count / new_count)
                    + batch_mean.to(self._device) * (num_tokens / new_count)
                )
                self._token_counts[layer_name] = new_count

            self._total_tokens_seen += num_tokens

        return hook_fn

    @property
    def enabled(self) -> bool:
        """Whether stats collection is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable stats collection without removing hooks."""
        if value and not self._enabled:
            logger.debug("ActivationStatsCollector: enabled")
        elif not value and self._enabled:
            logger.debug("ActivationStatsCollector: disabled")
        self._enabled = value

    @property
    def num_layers_tracked(self) -> int:
        """Number of layers with accumulated stats."""
        return len(self._running_mean)

    @property
    def total_tokens_seen(self) -> int:
        """Total tokens processed across all collection cycles."""
        return self._total_tokens_seen

    def mark_prompt_boundary(self) -> None:
        """Signal that a micro-batch (prompt group) has been processed.

        Call after each micro-batch forward to advance the sample counter.
        Once calibration_prompt_num batches are collected, sample storage stops
        but Welford running mean continues accumulating.
        """
        if self._sample_batches_collected < self._calibration_prompt_num:
            self._sample_batches_collected += 1

    def collect_and_clear(
        self,
    ) -> Optional[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        """Move collected stats and samples to CPU and reset buffers.

        Returns:
            Tuple of (stats_dict, samples_dict) where:
            - stats_dict: layer name → per-channel mean(|X|) [in_features] on CPU
            - samples_dict: layer name → activation sample [<=128, in_features] on CPU
            Returns None if no stats were collected.
        """
        if not self._running_mean:
            logger.debug("ActivationStatsCollector: no stats to collect")
            return None

        cpu_stats: dict[str, torch.Tensor] = {}
        for name, gpu_tensor in self._running_mean.items():
            cpu_stats[name] = gpu_tensor.cpu()

        # Concatenate sample lists into single tensors per layer.
        cpu_samples: dict[str, torch.Tensor] = {}
        for name, sample_list in self._activation_samples.items():
            if sample_list:
                cpu_samples[name] = torch.cat(sample_list, dim=0)

        num_layers = len(cpu_stats)
        sample_counts = list(self._token_counts.values())
        min_tokens = min(sample_counts) if sample_counts else 0
        max_tokens = max(sample_counts) if sample_counts else 0
        sample_tokens = sum(s.shape[0] for s in cpu_samples.values()) if cpu_samples else 0

        logger.info(
            "ActivationStatsCollector: %d layers, %d prompts sampled, "
            "%d sample tokens (tokens per layer: min=%d, max=%d, total_seen=%d)",
            num_layers,
            self._sample_batches_collected,
            sample_tokens,
            min_tokens,
            max_tokens,
            self._total_tokens_seen,
        )

        # Clear GPU buffers and samples (free memory).
        self._running_mean.clear()
        self._token_counts.clear()
        self._activation_samples.clear()
        self._sample_batches_collected = 0

        return cpu_stats, cpu_samples

    def remove_hooks(self) -> None:
        """Remove all registered forward pre-hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        logger.info("ActivationStatsCollector: removed all hooks")

    def __del__(self) -> None:
        """Cleanup hooks on garbage collection."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()
