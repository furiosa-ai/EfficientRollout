"""Toggle decision function and evaluation utilities.

Primary interface uses L_accept (expected accepted length).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SDToggleConfig
from .roofline import predict_speedup, compute_r, compute_v


def should_enable_sd(
    config: SDToggleConfig,
    B: int,
    S: int,
    gamma: int,
    L_accept: Optional[float] = None,
    margin: float = 0.0,
) -> bool:
    """Predict whether speculative decoding is beneficial.

    Args:
        config: Calibrated SD toggle configuration
        B: Batch size
        S: Sequence length (context)
        gamma: Number of draft tokens
        L_accept: Expected accepted length per SD cycle. If None, defaults to
            float(gamma) (trained-policy regime).
        margin: Safety margin (toggle at speedup >= 1.0 + margin)

    Returns:
        True if SD is predicted to be beneficial
    """
    if L_accept is None:
        L_accept = float(gamma)
    speedup = float(predict_speedup(B, S, gamma, L_accept, config))
    return speedup >= 1.0 + margin


def predict_decision(
    config: SDToggleConfig,
    B: int,
    S: int,
    gamma: int,
    L_accept: Optional[float] = None,
    margin: float = 0.0,
) -> dict:
    """Detailed prediction with all intermediate values.

    L_accept defaults to float(gamma) if None (trained-policy regime).

    Returns:
        Dict with keys: sd_on, speedup, r, v, L_accept, gamma,
        regime_T_T (mem/cmp), regime_T_V (mem/cmp)
    """
    if L_accept is None:
        L_accept = float(gamma)
    r = float(compute_r(B, S, gamma, config))
    v = float(compute_v(B, S, gamma, config))
    speedup = L_accept / (gamma * r + v)
    sd_on = speedup >= 1.0 + margin

    # Determine regimes
    cal = config.calibration
    gc = config.get_gamma_params(gamma)
    hw = config.hardware
    md = config.model
    M_t = (md.W_t + cal.kappa_eff * B * S) / hw.BW_eff
    C = (B * md.C_dense + B * S * md.C_attn) / cal.F_eff
    C_v = (B * (gamma + 1) * md.C_dense + B * S * (gamma + 1) * md.C_attn) / cal.F_eff

    return {
        "sd_on": sd_on,
        "speedup": round(speedup, 4),
        "r": round(r, 4),
        "v": round(v, 4),
        "L_accept": L_accept,
        "gamma": gamma,
        "B": B,
        "S": S,
        "regime_T_T": "mem" if M_t >= C else "cmp",
        "regime_T_V": "mem" if M_t >= C_v else "cmp",
    }


def predict_batch(
    config: SDToggleConfig,
    queries: list[tuple[int, int, int, float]],
    margin: float = 0.0,
) -> list[bool]:
    """Batch prediction for multiple (B, S, gamma, L_accept) queries.

    Args:
        config: Calibrated configuration
        queries: List of (B, S, gamma, L_accept) tuples
        margin: Safety margin

    Returns:
        List of bool decisions
    """
    return [
        should_enable_sd(config, B, S, gamma, L, margin)
        for B, S, gamma, L in queries
    ]


def evaluate_sign_accuracy(
    config: SDToggleConfig,
    csv_path: str | Path,
    L_accept: float,
    gammas: Optional[list[int]] = None,
    pow2_only: bool = False,
) -> float:
    """Evaluate sign accuracy: does the model correctly predict SD benefit/harm?

    Computes empirical speedup from CSV component data and compares the sign
    (beneficial vs harmful) with the model's prediction.

    Note: This measures component-ratio sign accuracy using r = T_D/T_T and
    v = T_V/T_T derived from raw component timings. It does NOT measure
    makespan-based sign accuracy. The component-ratio speedup systematically
    underpredicts actual makespan speedup by ~13% due to per-step autoregressive
    overhead amortization. Use this metric for model calibration quality;
    expect actual on-device gains to be moderately higher.

    Args:
        config: Calibrated configuration
        csv_path: Path to finegrain component sweep CSV
        L_accept: Expected accepted length
        gammas: Gamma values to evaluate (auto-detect if None)
        pow2_only: If True, only evaluate batch sizes that are powers of 2

    Returns:
        Sign accuracy as float in [0, 1]
    """
    from .fit import load_csv

    rows = load_csv(csv_path)

    # Build lookup: (batch, seq, gamma, mode) -> median_ms
    lookup: dict[tuple[int, int, int, str], float] = {}
    for r in rows:
        key = (r["batch"], r["seq"], r["gamma"], r["mode"])
        lookup[key] = r["median_ms"]

    # Determine gammas to evaluate
    if gammas is None:
        gammas = sorted(set(
            r["gamma"] for r in rows
            if r["gamma"] > 0 and r["mode"] == "drafter_decode"
        ))

    correct = 0
    total = 0

    for gamma in gammas:
        # Get unique (B, S) pairs for this gamma
        bs_pairs = sorted(set(
            (r["batch"], r["seq"])
            for r in rows
            if r["gamma"] == gamma and r["mode"] == "drafter_decode"
        ))

        pow2_set = {1, 2, 4, 8, 16, 32, 64, 128}
        for B, S in bs_pairs:
            if pow2_only and B not in pow2_set:
                continue
            # Need T_T (from gamma=0), T_D, T_V for empirical speedup
            T_T_key = (B, S, 0, "target_decode")
            T_D_key = (B, S, gamma, "drafter_decode")
            T_V_key = (B, S, gamma, "target_verify")

            if T_T_key not in lookup or T_D_key not in lookup or T_V_key not in lookup:
                continue

            T_T_ms = lookup[T_T_key]
            T_D_ms = lookup[T_D_key]
            T_V_ms = lookup[T_V_key]

            # Empirical speedup: L_accept / (gamma * r_emp + v_emp)
            if T_T_ms <= 0:
                continue
            r_emp = T_D_ms / T_T_ms
            v_emp = T_V_ms / T_T_ms
            denom = gamma * r_emp + v_emp
            if denom <= 0:
                continue
            empirical_speedup = L_accept / denom

            # Model prediction
            predicted_speedup = float(predict_speedup(B, S, gamma, L_accept, config))

            # Sign comparison
            emp_beneficial = empirical_speedup > 1.0
            pred_beneficial = predicted_speedup > 1.0

            if emp_beneficial == pred_beneficial:
                correct += 1
            total += 1

    if total == 0:
        warnings.warn("No valid data points for sign accuracy evaluation", stacklevel=2)
        return 0.0

    return correct / total
