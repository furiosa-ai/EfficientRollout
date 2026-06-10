"""Core roofline model: T_T, T_D, T_V, speedup formulas.

All times in seconds. Overhead terms c_T/c_D/c_V in microseconds per batch element.
"""
from __future__ import annotations

import numpy as np
from typing import Union

from .config import SDToggleConfig

Numeric = Union[float, np.ndarray]


def predict_T_T(
    B: Numeric, S: Numeric,
    W_t: float, kappa_eff: float, BW_eff: float,
    C_dense: float, C_attn: float, F_eff: float,
    c_T: float = 0.0,
    c_comm: float = 0.0,
) -> Numeric:
    """Predict target decode latency (seconds).

    T_T = max(M_t, C) + c_T * B * 1e-6 + c_comm

    Args:
        B: Batch size
        S: Sequence length (context)
        W_t: Target model weight bytes (per-GPU, already /tp)
        kappa_eff: Effective KV cache bytes per (B*S) (per-GPU, already /tp)
        BW_eff: Effective memory bandwidth (bytes/s, per-GPU)
        C_dense: Per-token dense compute (FLOPS, per-GPU, already /tp)
        C_attn: Per-token-per-context attention compute (FLOPS, per-GPU, already /tp)
        F_eff: Effective compute throughput (FLOPS, per-GPU)
        c_T: Target batch overhead (microseconds per batch element)
        c_comm: Pre-measured NCCL overhead per forward pass (seconds); 0 for tp=1

    Returns:
        Predicted latency in seconds
    """
    M_t = (W_t + kappa_eff * B * S) / BW_eff
    C = (B * C_dense + B * S * C_attn) / F_eff
    return np.maximum(M_t, C) + c_T * B * 1e-6 + c_comm


def predict_T_D(
    B: Numeric, S: Numeric,
    W_d: float, eta_d: float, kappa_eff: float, BW_eff: float,
    C_dense: float, C_attn: float, F_eff: float,
    c_D: float = 0.0,
    c_comm: float = 0.0,
) -> Numeric:
    """Predict drafter decode latency (seconds).

    T_D = max(M_d, C) + c_D * B * 1e-6 + c_comm

    Args:
        B: Batch size
        S: Sequence length
        W_d: Drafter model weight bytes (per-GPU, already /tp)
        eta_d: Drafter overhead multiplier (>= 1.0, captures Marlin backend overhead)
        kappa_eff: Effective KV cache bytes per (B*S) (per-GPU, already /tp)
        BW_eff: Effective memory bandwidth (bytes/s, per-GPU)
        C_dense: Per-token dense compute (FLOPS, per-GPU, already /tp)
        C_attn: Per-token-per-context attention compute (FLOPS, per-GPU, already /tp)
        F_eff: Effective compute throughput (FLOPS, per-GPU)
        c_D: Drafter batch overhead (microseconds per batch element)
        c_comm: Pre-measured NCCL overhead per forward pass (seconds); 0 for tp=1

    Returns:
        Predicted latency in seconds
    """
    M_d = (eta_d * W_d + kappa_eff * B * S) / BW_eff
    C = (B * C_dense + B * S * C_attn) / F_eff
    return np.maximum(M_d, C) + c_D * B * 1e-6 + c_comm


def predict_T_V(
    B: Numeric, S: Numeric, gamma: int,
    W_t: float, kappa_eff: float, BW_eff: float,
    C_dense: float, C_attn: float, F_eff: float,
    c_V: float = 0.0,
    c_comm: float = 0.0,
    beta: float = 0.0,
) -> Numeric:
    """Predict verify latency (seconds).

    T_V = max(M_V, (gamma+1)*C) + c_V * B * 1e-6 + c_comm

    where M_V = [W_t + κ·B·S·(1 + β·γ)] / BW_eff.

    The β term captures q-dependent increase in effective memory traffic
    during verify, attributable to query-tiling-induced KV re-access and
    activation traffic in FlashAttention. β is a hardware/kernel property
    measured once per GPU, shared across models.

    When β=0 (default), this reduces to the original form M_V = M_T.

    The (gamma+1) multiplier on compute reflects verifying gamma draft tokens
    plus one bonus token. phi(gamma) = 1 (FlashAttention fully reuses KV).

    Args:
        B: Batch size
        S: Sequence length (context)
        gamma: Number of draft tokens
        W_t: Target model weight bytes (per-GPU, already /tp)
        kappa_eff: Effective KV cache bytes per (B*S) (per-GPU, already /tp)
        BW_eff: Effective memory bandwidth (bytes/s, per-GPU)
        C_dense: Per-token dense compute (FLOPS, per-GPU, already /tp)
        C_attn: Per-token-per-context attention compute (FLOPS, per-GPU, already /tp)
        F_eff: Effective compute throughput (FLOPS, per-GPU)
        c_V: Verify batch overhead (microseconds per batch element)
        c_comm: Pre-measured NCCL overhead per forward pass (seconds); 0 for tp=1
        beta: Verify memory traffic scaling factor (hardware property, >= 0)

    Returns:
        Predicted latency in seconds
    """
    M_V = (W_t + kappa_eff * B * S * (1.0 + beta * gamma)) / BW_eff
    C_verify = (B * (gamma + 1) * C_dense + B * S * (gamma + 1) * C_attn) / F_eff
    return np.maximum(M_V, C_verify) + c_V * B * 1e-6 + c_comm


def compute_r(
    B: Numeric, S: Numeric, gamma: int,
    config: SDToggleConfig,
) -> Numeric:
    """Compute r = T_D / T_T ratio."""
    cal = config.calibration
    gc = config.get_gamma_params(gamma)
    hw = config.hardware
    md = config.model

    t_t = predict_T_T(B, S, md.W_t, cal.kappa_eff, hw.BW_eff,
                       md.C_dense, md.C_attn, cal.F_eff, gc.c_T, hw.c_comm)
    t_d = predict_T_D(B, S, md.W_d, cal.eta_d, cal.kappa_eff, hw.BW_eff,
                       md.C_dense, md.C_attn, cal.F_eff, gc.c_D, hw.c_comm)
    return t_d / t_t


def compute_v(
    B: Numeric, S: Numeric, gamma: int,
    config: SDToggleConfig,
) -> Numeric:
    """Compute v = T_V / T_T ratio."""
    cal = config.calibration
    gc = config.get_gamma_params(gamma)
    hw = config.hardware
    md = config.model

    t_t = predict_T_T(B, S, md.W_t, cal.kappa_eff, hw.BW_eff,
                       md.C_dense, md.C_attn, cal.F_eff, gc.c_T, hw.c_comm)
    t_v = predict_T_V(B, S, gamma, md.W_t, cal.kappa_eff, hw.BW_eff,
                       md.C_dense, md.C_attn, cal.F_eff, gc.c_V, hw.c_comm,
                       beta=cal.beta)
    return t_v / t_t


def predict_speedup(
    B: Numeric, S: Numeric, gamma: int,
    L_accept: float | None,
    config: SDToggleConfig,
) -> Numeric:
    """Predict SD speedup over autoregressive decoding.

    Speedup = L_accept / (gamma * r + v)

    where r = T_D/T_T, v = T_V/T_T.

    Args:
        B: Batch size
        S: Sequence length
        gamma: Number of draft tokens
        L_accept: Expected accepted length per SD cycle. If None, defaults to
            float(gamma) (trained-policy regime assumption).
        config: Calibrated SD toggle configuration

    Returns:
        Predicted speedup (>1.0 means SD is beneficial)
    """
    if L_accept is None:
        L_accept = float(gamma)
    r = compute_r(B, S, gamma, config)
    v = compute_v(B, S, gamma, config)
    denom = gamma * r + v
    if isinstance(denom, np.ndarray):
        denom = np.maximum(denom, 1e-12)
    elif denom < 1e-12:
        return np.inf
    return L_accept / denom


def find_boundary_B(
    S: float, gamma: int, L_accept: float,
    config: SDToggleConfig,
    B_range: tuple[float, float] = (1.0, 128.0),
) -> float | None:
    """Find the batch size B* where speedup = 1.0 for given (S, gamma, L_accept).

    Uses bisection (scipy.optimize.brentq) since the roofline max() prevents
    analytic solution.

    Returns:
        B* (float) or None if speedup is always >1 or always <1 in range
    """
    from scipy.optimize import brentq

    def f(B: float) -> float:
        return float(predict_speedup(B, S, gamma, L_accept, config)) - 1.0

    lo, hi = B_range
    f_lo, f_hi = f(lo), f(hi)

    # No crossing — speedup is same sign at both ends
    if f_lo * f_hi > 0:
        return None

    try:
        return brentq(f, lo, hi, xtol=0.1)
    except ValueError:
        return None
