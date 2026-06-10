"""AWQ beta search for activation-aware weight quantization.

Per-layer optimal beta search using block output MSE (when activation
samples available) or weight MSE fallback. Uses stale activation
statistics from FSDP compute_log_prob() to compute per-channel AWQ
scales that minimize quantization error.

Scale absorption: for layers with preceding LayerNorm (q_proj, gate_proj),
AWQ scales are absorbed by dividing LN weights (see
_absorb_awq_scales_into_layernorms in quant_self_proposer.py).
For down_proj, FC-FC absorption applies scales to the up portion of the
fused gate_up_proj predecessor. o_proj has no absorption → weight MSE only.
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _batch_pseudo_quantize(
    w_batch: torch.Tensor, bits: int = 4, group_size: int = 128,
) -> torch.Tensor:
    """Asymmetric pseudo-quantize for batched weights [C, K, N].

    Matches Marlin kernel's asymmetric [0, 2^bits-1] with zero_point.
    Groups along K dimension.
    """
    C, K, N = w_batch.shape
    max_q = 2 ** bits - 1  # 15 for 4-bit

    if group_size <= 0:
        group_size = K

    assert K % group_size == 0, (
        f"in_features ({K}) must be divisible by group_size ({group_size})"
    )

    # Group along K: [C, K//g, g, N]
    w_grouped = w_batch.reshape(C, K // group_size, group_size, N)

    max_val = w_grouped.amax(dim=2, keepdim=True)   # [C, K//g, 1, N]
    min_val = w_grouped.amin(dim=2, keepdim=True)    # [C, K//g, 1, N]

    scale = (max_val - min_val).clamp(min=1e-5) / max_q
    zp = (-torch.round(min_val / scale)).clamp(0, max_q).int()

    w_q = torch.round(w_grouped / scale).int() + zp
    w_q = w_q.clamp(0, max_q)
    w_ref = (w_q - zp).to(w_batch.dtype) * scale

    return w_ref.reshape(C, K, N)


def search_beta_per_layer(
    fp16_weight: torch.Tensor,
    act_stats: torch.Tensor,
    activation_sample: torch.Tensor | None = None,
    group_size: int = 128,
    num_candidates: int = 20,
) -> tuple[float, torch.Tensor]:
    """Search for optimal AWQ beta for a single layer.

    All candidates are evaluated in a single batched GPU operation:
    - Scales for all betas computed in parallel
    - Batch pseudo-quantize: [C, K, N] in one kernel
    - Batch MSE: bmm for block MSE, or vectorized for weight MSE

    Args:
        fp16_weight: [out_features, in_features] FP16 weight (PyTorch convention)
        act_stats: [in_features] per-channel mean |activation|
        activation_sample: Optional [num_tokens, in_features] activation sample.
            When provided, uses block output MSE instead of weight MSE.
        group_size: Quantization group size (default 128)
        num_candidates: Number of beta values to evaluate (default 20)

    Returns:
        (best_beta, best_scales) where best_scales is [in_features] on same device
    """
    # Transpose to Marlin convention: [in_features, out_features]
    w = fp16_weight.t().contiguous().float()
    K, N = w.shape

    act_stats = act_stats.to(device=w.device, dtype=torch.float32)
    C = num_candidates

    # --- All betas and scales at once ---
    betas = torch.linspace(0, 1, C, device=w.device)  # [C]
    # all_scales: [C, K]
    all_scales = act_stats.unsqueeze(0).pow(betas.unsqueeze(1)).clamp(min=1e-5)
    # Geometric-mean normalization per candidate
    all_scales = all_scales / all_scales.log().mean(dim=1, keepdim=True).exp()

    # --- Batch scale + quantize ---
    # w_scaled: [C, K, N]
    # TODO(7B): For large models (e.g. 7B gate_up K=3584,N=18944), this
    # allocates ~5GB per [C,K,N] tensor. Peak ~10GB with w_ref. If OOM,
    # chunk candidates into groups of 5 and loop.
    w_scaled = w.unsqueeze(0) * all_scales.unsqueeze(2)
    # Batch pseudo-quantize (asymmetric, matching Marlin kernel)
    w_ref = _batch_pseudo_quantize(w_scaled, bits=4, group_size=group_size)

    # --- Batch MSE ---
    if activation_sample is not None:
        # Block output MSE with scale absorption
        x = activation_sample.to(device=w.device, dtype=torch.float32)  # [T, K]
        org_out = x @ w  # [T, N] — reference (unscaled)

        # x_scaled: [C, T, K] — absorption: x / scales
        x_scaled = x.unsqueeze(0) / all_scales.unsqueeze(1)
        # awq_out: [C, T, N] — batched matmul
        awq_out = torch.bmm(x_scaled, w_ref)
        # errors: [C]
        errors = (org_out.unsqueeze(0) - awq_out).pow(2).mean(dim=(1, 2))
    else:
        # Weight MSE fallback
        w_deq = w_ref / all_scales.unsqueeze(2)
        errors = (w.unsqueeze(0) - w_deq).pow(2).mean(dim=(1, 2))

    best_idx = errors.argmin().item()
    best_beta = betas[best_idx].item()
    best_scales = all_scales[best_idx]

    return best_beta, best_scales


def compute_awq_scales_all_layers(
    target_model: nn.Module,
    stale_stats: dict[str, torch.Tensor],
    stale_samples: dict[str, torch.Tensor] | None = None,
    group_size: int = 128,
    num_candidates: int = 20,
) -> dict[str, torch.Tensor]:
    """Compute optimal AWQ scales for all layers with activation stats.

    Maps collector layer names (e.g. "model.layers.0.self_attn.q_proj")
    to target model weight keys (e.g. "model.layers.0.self_attn.q_proj.weight")
    and runs per-layer beta search.

    Args:
        target_model: Target model with FP16 weights
        stale_stats: dict from ActivationStatsCollector.collect_and_clear()
            Keys are module names, values are [in_features] tensors on CPU
        stale_samples: Optional dict of activation samples [<=128, in_features]
            on CPU. When provided, block output MSE is used instead of weight MSE.
        group_size: Quantization group size (default 128)
        num_candidates: Number of beta candidates per layer (default 20)

    Returns:
        dict mapping weight_key → optimal scales tensor on GPU
    """
    t0 = time.perf_counter()
    target_params = dict(target_model.named_parameters())
    if stale_samples is None:
        stale_samples = {}

    scales_dict: dict[str, torch.Tensor] = {}
    betas: dict[str, float] = {}
    skipped = 0

    # Map FSDP unfused names → vLLM fused names for weight lookup.
    # FSDP/HF model has: q_proj, k_proj, v_proj, gate_proj, up_proj
    # vLLM model has: qkv_proj (fused q+k+v), gate_up_proj (fused gate+up)
    # We search on unfused stats but store scales keyed by FSDP name
    # (the proposer's _resolve_awq_scales handles the mapping to fused names).
    for layer_name, act_stats in stale_stats.items():
        # Try direct lookup first (unfused model or o_proj/down_proj)
        weight_key = f"{layer_name}.weight"
        fp16_weight = None

        if weight_key in target_params:
            fp16_weight = target_params[weight_key].data
        else:
            # Fused weight: extract the relevant shard
            layer_prefix = layer_name.rsplit(".", 1)[0]  # e.g. "model.layers.0.self_attn"
            suffix = layer_name.rsplit(".", 1)[1]  # e.g. "q_proj"

            if suffix in ("q_proj", "k_proj", "v_proj"):
                fused_key = f"{layer_prefix}.qkv_proj.weight"
                if fused_key in target_params:
                    fused_w = target_params[fused_key].data
                    # qkv_proj: [q_size + kv_size + kv_size, hidden]
                    # Shard sizes from weight shape and act_stats dim
                    shard_size = act_stats.shape[0]  # in_features = this proj's output size
                    if suffix == "q_proj":
                        fp16_weight = fused_w[:shard_size]
                    elif suffix == "k_proj":
                        q_size = fused_w.shape[0] - 2 * shard_size  # total - 2*kv_size
                        # Actually: act_stats for k_proj has shape [kv_dim]
                        # but we need q_size to find k's offset
                        # Safer: use the stats to find q_proj's size
                        q_stats_key = layer_name.replace("k_proj", "q_proj")
                        q_size = stale_stats[q_stats_key].shape[0] if q_stats_key in stale_stats else (fused_w.shape[0] - 2 * shard_size)
                        fp16_weight = fused_w[q_size:q_size + shard_size]
                    else:  # v_proj
                        q_stats_key = layer_name.replace("v_proj", "q_proj")
                        q_size = stale_stats[q_stats_key].shape[0] if q_stats_key in stale_stats else (fused_w.shape[0] - 2 * shard_size)
                        fp16_weight = fused_w[q_size + shard_size:q_size + 2 * shard_size]

            elif suffix in ("gate_proj", "up_proj"):
                fused_key = f"{layer_prefix}.gate_up_proj.weight"
                if fused_key in target_params:
                    fused_w = target_params[fused_key].data
                    half = fused_w.shape[0] // 2
                    if suffix == "gate_proj":
                        fp16_weight = fused_w[:half]
                    else:  # up_proj
                        fp16_weight = fused_w[half:]

        if fp16_weight is None:
            skipped += 1
            logger.debug(
                "AWQ search: no weight found for %s, skipping", layer_name,
            )
            continue

        # Block output MSE for layers with ANY absorption (LN or FC-FC).
        # q_proj, gate_proj: LN absorption
        # down_proj: FC-FC absorption
        # o_proj, k_proj, v_proj, up_proj: weight MSE (shared scales from q/gate)
        has_absorption = any(
            layer_name.endswith(s) for s in ("q_proj", "gate_proj", "down_proj")
        )
        activation_sample = (
            stale_samples.get(layer_name, None)
            if has_absorption else None
        )
        beta, scales = search_beta_per_layer(
            fp16_weight, act_stats,
            activation_sample=activation_sample,
            group_size=group_size,
            num_candidates=num_candidates,
        )

        # Store scales keyed by FSDP name (unfused).
        # The proposer's _resolve_awq_scales() maps these to drafter modules.
        scales_dict[weight_key] = scales
        betas[weight_key] = beta

    elapsed = time.perf_counter() - t0

    # Log timing and beta distribution per layer type
    if betas:
        beta_by_type: dict[str, list[float]] = {}
        for key, beta in betas.items():
            # Extract layer type: e.g. "q_proj" from "model.layers.0.self_attn.q_proj.weight"
            parts = key.rsplit(".", 2)  # [..., "q_proj", "weight"]
            layer_type = parts[-2] if len(parts) >= 2 else "unknown"
            beta_by_type.setdefault(layer_type, []).append(beta)

        type_summary = ", ".join(
            f"{lt}: mean={sum(bs)/len(bs):.3f} [{min(bs):.2f}-{max(bs):.2f}]"
            for lt, bs in sorted(beta_by_type.items())
        )
        logger.info(
            "AWQ beta search: %d layers in %.3fs (skipped %d). "
            "Beta distribution: %s",
            len(scales_dict), elapsed, skipped, type_summary,
        )

    return scales_dict
