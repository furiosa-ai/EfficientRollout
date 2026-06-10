"""Deterministic model constants derived from HF architecture configs.

All values in base units: bytes (weights, KV cache), FLOPS (compute).
No GPU or torch dependency — pure computation from architecture parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConstants:
    """Deterministic constants for a (model, tp, quant_ratio) configuration."""
    name: str
    tp: int
    quant_ratio: float  # drafter weight quantization ratio (e.g. 0.25 for W4)

    # Architecture
    D: int          # hidden dimension
    D_ff: int       # FFN intermediate dimension
    L: int          # number of transformer layers
    n_heads: int    # number of attention heads
    n_kv: int       # number of KV heads
    d_h: int        # head dimension
    V: int          # vocabulary size
    gqa: int        # grouped query attention factor (n_heads / n_kv)

    # Derived (per-GPU after TP split)
    W_t: float      # target model weight bytes (FP16)
    W_d: float      # drafter model weight bytes (quantized)
    rho: float      # W_d / W_t ratio
    kappa_theoretical: int  # KV cache bytes per (B*S) unit
    C_dense: float  # per-token dense compute (FLOPS)
    C_attn: float   # per-token-per-context attention compute (FLOPS)


# ── Model architecture registry ──
# Values from HF configs; compute logic matches scripts/lv1_model_constants.py

_MODEL_ARCHS = {
    "Qwen2.5-7B": {
        "D": 3584, "D_ff": 18944, "L": 28,
        "n_heads": 28, "n_kv": 4, "d_h": 128,
        "V": 152064,
    },
    "Qwen/Qwen2.5-7B": {  # alias with HF prefix
        "D": 3584, "D_ff": 18944, "L": 28,
        "n_heads": 28, "n_kv": 4, "d_h": 128,
        "V": 152064,
    },
    "LLaMA3.1-8B": {
        "D": 4096, "D_ff": 14336, "L": 32,
        "n_heads": 32, "n_kv": 8, "d_h": 128,
        "V": 128256,
    },
    "meta-llama/Meta-Llama-3.1-8B": {  # alias
        "D": 4096, "D_ff": 14336, "L": 32,
        "n_heads": 32, "n_kv": 8, "d_h": 128,
        "V": 128256,
    },
    "Llama-3.1-8B-Instruct": {  # same architecture as Base
        "D": 4096, "D_ff": 14336, "L": 32,
        "n_heads": 32, "n_kv": 8, "d_h": 128,
        "V": 128256,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {  # alias
        "D": 4096, "D_ff": 14336, "L": 32,
        "n_heads": 32, "n_kv": 8, "d_h": 128,
        "V": 128256,
    },
    "Qwen2.5-14B": {
        "D": 5120, "D_ff": 13824, "L": 48,
        "n_heads": 40, "n_kv": 8, "d_h": 128,
        "V": 152064,
    },
    "Qwen/Qwen2.5-14B": {  # alias
        "D": 5120, "D_ff": 13824, "L": 48,
        "n_heads": 40, "n_kv": 8, "d_h": 128,
        "V": 152064,
    },
}

# ── GPU specs ──
GPU_SPECS = {
    "A100-SXM4-80GB": {"BW_peak": 2.039e12, "F_peak": 312e12},
    "A100": {"BW_peak": 2.039e12, "F_peak": 312e12},  # alias
}


def _normalize_model_name(name: str) -> str:
    """Try to resolve model name to a known key."""
    if name in _MODEL_ARCHS:
        return name
    # Strip common prefixes
    for prefix in ("Qwen/", "meta-llama/", "Meta-Llama-"):
        stripped = name.replace(prefix, "")
        if stripped in _MODEL_ARCHS:
            return stripped
    # Try matching by suffix (require at least 5 chars to avoid false matches like "8B")
    if len(name) >= 5:
        for key in _MODEL_ARCHS:
            if name.endswith(key) or key.endswith(name):
                return key
    raise ValueError(
        f"Unknown model: {name}. Known models: "
        f"{[k for k in _MODEL_ARCHS if '/' not in k]}"
    )


def compute_constants(
    model_name: str,
    tp: int = 1,
    quant_ratio: float = 0.25,
) -> ModelConstants:
    """Compute deterministic model constants from architecture parameters.

    Args:
        model_name: Model identifier (e.g. "Qwen2.5-7B", "Qwen/Qwen2.5-7B")
        tp: Tensor parallelism degree
        quant_ratio: Drafter weight quantization ratio (0.25 = W4, 0.5 = W8)

    Returns:
        ModelConstants with all derived values in base units (bytes, FLOPS)
    """
    key = _normalize_model_name(model_name)
    arch = _MODEL_ARCHS[key]

    D = arch["D"]
    D_ff = arch["D_ff"]
    L = arch["L"]
    n_heads = arch["n_heads"]
    n_kv = arch["n_kv"]
    d_h = arch["d_h"]
    V = arch["V"]
    gqa = n_heads // n_kv

    # ── Per-layer params (matching reference lv1_model_constants.py) ──
    # Attention: Q(D * n_heads*d_h) + K(D * n_kv*d_h) + V(D * n_kv*d_h) + O(n_heads*d_h * D)
    attn_params_per_layer = (
        D * (n_heads * d_h) +   # Q
        D * (n_kv * d_h) +      # K
        D * (n_kv * d_h) +      # V
        (n_heads * d_h) * D     # O
    )
    # FFN: SwiGLU = gate(D*D_ff) + up(D*D_ff) + down(D_ff*D)
    ffn_params_per_layer = D * D_ff * 2 + D_ff * D  # gate + up + down

    transformer_params = L * (attn_params_per_layer + ffn_params_per_layer)
    embed_params = V * D
    lm_head_params = V * D
    ln_params = (2 * L + 1) * D  # 2 per layer + 1 final norm

    total_params = transformer_params + embed_params + lm_head_params + ln_params

    # Total weight bytes (FP16), divided by TP (all params split)
    W_t = total_params * 2 / tp

    # Drafter: quantized linear layers + FP16 embed/LM head/LN
    quantized_params = transformer_params  # only linear layers quantized
    fp16_params = embed_params + lm_head_params + ln_params
    W_d = (quantized_params * 2 * quant_ratio + fp16_params * 2) / tp

    rho = W_d / W_t

    # ── KV cache bytes per (B*S) unit, PER GPU ──
    # Per token across all layers: K + V, each [n_kv, d_h] in FP16.
    # KV cache is sharded along the n_kv dimension (Megatron-LM style), so
    # each GPU stores n_kv/tp heads. When n_kv < tp, KV heads are replicated
    # on the excess ranks — use max(n_kv // tp, 1) to clamp. This keeps
    # kappa_theoretical as a true per-GPU reference so kappa_ratio stays
    # in a consistent [0.4, 1.0] range across tp values.
    n_kv_per_gpu = max(n_kv // tp, 1)
    kappa = L * 2 * n_kv_per_gpu * d_h * 2

    # ── Compute per token (FLOPS) ──
    # Attention linear per layer: 2 * (Q + K + V + O) matmul FLOPs
    attn_linear_flops = 2 * attn_params_per_layer
    # FFN per layer: 2 * (gate + up + down)
    ffn_flops = 2 * ffn_params_per_layer
    # LM head
    lm_head_flops = 2 * D * V
    # Divided by TP: each GPU computes 1/tp of the total
    C_dense = (L * (attn_linear_flops + ffn_flops) + lm_head_flops) // tp

    # Attention compute per token per context position
    # QK^T + AV: per head, per token, per context: 2*d_h each
    # Divided by TP: attention heads are split across GPUs
    C_attn = L * n_heads * 2 * d_h * 2 // tp

    # Use short name (without HF prefix)
    short_name = key

    return ModelConstants(
        name=short_name,
        tp=tp,
        quant_ratio=quant_ratio,
        D=D, D_ff=D_ff, L=L,
        n_heads=n_heads, n_kv=n_kv, d_h=d_h,
        V=V, gqa=gqa,
        W_t=W_t, W_d=W_d, rho=rho,
        kappa_theoretical=kappa,
        C_dense=C_dense, C_attn=C_attn,
    )


def list_models() -> list[str]:
    """Return list of known model names (without HF prefix aliases)."""
    return [k for k in _MODEL_ARCHS if "/" not in k]
