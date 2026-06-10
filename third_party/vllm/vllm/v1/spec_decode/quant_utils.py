"""Quantization utilities for online W4 Marlin-based self-speculative decoding.

This module provides:
- quantize_to_marlin: FP16 weight -> int4 -> AWQ pack -> Marlin repack pipeline

All internal helper functions (_torch_awq_pack, _torch_marlin_permute_scales, etc.)
are GPU-only torch ops — no numpy dependency.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights,
)
from vllm.scalar_type import scalar_types

logger = logging.getLogger(__name__)

# Constants
NUM_BITS = 4
PACK_FACTOR = 32 // NUM_BITS  # 8
MARLIN_TILE = 16
QUANT_TYPE = scalar_types.uint4

# Pre-computed permutation indices (Python lists → CPU tensors cached once)
_AWQ_INTERLEAVE_4BIT = [0, 2, 4, 6, 1, 3, 5, 7]
_SCALE_PERM = [i + 8 * j for i in range(8) for j in range(8)]  # 64 elems
_SCALE_PERM_SINGLE = [
    2 * i + j for i in range(4) for j in [0, 1, 8, 9, 16, 17, 24, 25]
]  # 32 elems

# CPU-cached tensors (immune to CuMem sleep/wake GPU memory recycling).
# GPU cache was corrupted by CuMem level 2 sleep which zeroes all GPU memory.
# CPU tensors are transferred to GPU on each call — negligible overhead for
# 64 int64 values (512 bytes).
_cpu_cached_perms: dict[str, torch.Tensor] | None = None


def _get_perms(device: torch.device) -> dict[str, torch.Tensor]:
    """Get permutation tensors on the given device (cached on CPU)."""
    global _cpu_cached_perms
    if _cpu_cached_perms is None:
        _cpu_cached_perms = {
            "awq_interleave": torch.tensor(
                _AWQ_INTERLEAVE_4BIT, dtype=torch.long
            ),
            "scale_perm": torch.tensor(
                _SCALE_PERM, dtype=torch.long
            ),
            "scale_perm_single": torch.tensor(
                _SCALE_PERM_SINGLE, dtype=torch.long
            ),
        }
    return {k: v.to(device) for k, v in _cpu_cached_perms.items()}


def _torch_pack_cols(q_w: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Pack columns: 8 int4 values → 1 int32. Pure GPU torch ops."""
    size_k, size_n = q_w.shape
    pack_factor = 32 // num_bits
    q_w = q_w.to(torch.int32)
    packed = torch.zeros(
        size_k, size_n // pack_factor, dtype=torch.int32, device=q_w.device
    )
    for i in range(pack_factor):
        packed.bitwise_or_(q_w[:, i::pack_factor] << (num_bits * i))
    return packed


def _torch_awq_pack(q_w: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """AWQ interleave + column pack. Pure GPU torch ops."""
    size_k, size_n = q_w.shape
    perms = _get_perms(q_w.device)
    interleave = perms["awq_interleave"]

    # Interleave columns within groups of 8
    q_w = (
        q_w.reshape(-1, len(interleave))
        .index_select(1, interleave)
        .reshape(size_k, size_n)
        .contiguous()
    )
    return _torch_pack_cols(q_w, num_bits)


def _torch_marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    """Marlin scale permutation. Pure GPU torch ops."""
    perms = _get_perms(s.device)
    if group_size < size_k and group_size != -1:
        perm = perms["scale_perm"]
    else:
        perm = perms["scale_perm_single"]
    s = s.reshape(-1, len(perm)).index_select(1, perm)
    return s.reshape(-1, size_n).contiguous()


def _torch_marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int = 4
) -> torch.Tensor:
    """Marlin zero-point permutation + packing. Pure GPU torch ops."""
    perms = _get_perms(zp.device)
    scale_perm = perms["scale_perm"]
    interleave = perms["awq_interleave"]

    # Scale permutation
    zp = (
        zp.to(torch.int32)
        .reshape(-1, len(scale_perm))
        .index_select(1, scale_perm)
        .reshape(-1, size_n)
        .contiguous()
    )
    # AWQ interleave
    zp = (
        zp.reshape(-1, len(interleave))
        .index_select(1, interleave)
        .reshape(-1, size_n)
        .contiguous()
    )
    # Pack
    return _torch_pack_cols(zp, num_bits)


def quantize_to_marlin(
    fp16_weight: torch.Tensor,
    group_size: int = 128,
    scales_override: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize FP16 weight to Marlin-packed int4 format. GPU-only pipeline.

    Args:
        fp16_weight: [out_features, in_features] FP16 weight tensor
            (standard PyTorch nn.Linear convention)
        group_size: Quantization group size (default 128)
        scales_override: Optional per-channel scaling (for activation-aware quantization)

    Returns:
        (marlin_qweight, marlin_scales, marlin_qzeros) ready for Marlin kernel
    """
    # Transpose from PyTorch [out, in] to Marlin [in, out] convention
    w = fp16_weight.t().contiguous().float()
    size_k, size_n = w.shape  # size_k=in_features, size_n=out_features

    # Apply activation-aware scaling if provided (Tier 1/2)
    if scales_override is not None:
        w = w * scales_override.unsqueeze(1)

    # Step 1: Quantize to int4 with zero points (torch GPU ops)
    _w_ref, w_q, w_s, w_zp = quantize_weights(
        w, quant_type=QUANT_TYPE, group_size=group_size, zero_points=True
    )

    num_groups = size_k // group_size

    # Step 2: AWQ interleave + pack (GPU torch ops, replaces numpy awq_pack)
    qweight_awq = _torch_awq_pack(w_q, NUM_BITS)

    # Step 3: Repack AWQ → Marlin tiled format (CUDA kernel, already fast)
    marlin_qweight = ops.awq_marlin_repack(
        qweight_awq, size_k=size_k, size_n=size_n, num_bits=NUM_BITS
    )

    # Step 4: Permute scales for Marlin layout (GPU torch ops, float16)
    marlin_scales = _torch_marlin_permute_scales(
        w_s.to(torch.float16), size_k=size_k, size_n=size_n, group_size=group_size
    )

    # Step 5: Permute + pack zero-points for Marlin layout (GPU torch ops)
    marlin_qzeros = _torch_marlin_zero_points(
        w_zp, size_k=num_groups, size_n=size_n, num_bits=NUM_BITS
    )

    return marlin_qweight, marlin_scales, marlin_qzeros
