"""Staged calibration pipeline: CSV data → fitted params → SDToggleConfig.

3-stage pipeline, all 6 params are γ-independent:

Stage 0: (Optional) Load F_eff from GEMM micro-benchmark (hardware constant).
Stage 1: Fit (BW_eff, κ, c_T) from T_T (target_decode, gamma=0) data.
Stage 2: Fit (η_d, c_D, c_V) shared across ALL gammas simultaneously,
         using hybrid loss = 0.5·component_logMSE + 0.5·lv2_primitive.

β=0 fixed (tried, marginal gain, dropped).
F_eff from GEMM bench (212T for A100) or explicit argument.
"""
from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .config import (
    SDToggleConfig, HardwareConfig, ModelConfig,
    CalibrationConfig,
)
from .constants import compute_constants, GPU_SPECS, ModelConstants
from .roofline import predict_T_T, predict_T_D, predict_T_V


def load_F_eff_from_bench(bench_path: str | Path) -> float:
    """Load hardware F_eff from GEMM micro-benchmark JSON (Stage 0).

    Reads the ``F_eff_hardware_tflops`` field produced by
    ``scripts/profiling/measure_gemm_effective_tflops.py`` and converts
    TFLOPS → FLOPS.  This is the cross-model median throughput measured
    at saturation-regime batch sizes (M >= 256), representing the GPU's
    realized fp16 tensor-core throughput when GEMMs are compute-bound.

    Args:
        bench_path: Path to the bench JSON (e.g. ``sd_toggle/configs/F_eff_bench.json``)

    Returns:
        F_eff in FLOPS (e.g. 200e12 for 200 TFLOPS)

    Raises:
        FileNotFoundError: bench JSON does not exist
        ValueError: bench JSON missing saturation data
    """
    bench_path = Path(bench_path)
    with open(bench_path) as f:
        data = json.load(f)

    hw_tflops = data.get("F_eff_hardware_tflops")
    if hw_tflops is None:
        raise ValueError(
            f"No F_eff_hardware_tflops in {bench_path}. "
            "Re-run measure_gemm_effective_tflops.py with batch sizes >= 256."
        )

    F_eff = hw_tflops * 1e12
    print(f"[Stage 0] F_eff = {hw_tflops:.1f} TFLOPS (from {bench_path.name})")
    return F_eff


def load_c_comm_from_bench(
    bench_path: str | Path, model_name: str, tp: int = 2
) -> float:
    """Load per-model NCCL c_comm from allreduce micro-benchmark JSON (Stage 0).

    Reads ``c_comm_per_model_tp<tp>[<model_name>_tp<tp>].c_comm_B32_sec`` from
    the JSON produced by ``scripts/profiling/measure_nccl_allreduce.py``.
    B=32 is the toggle-transition-zone representative batch.

    Args:
        bench_path: Path to the bench JSON (e.g. ``sd_toggle/configs/c_comm_bench.json``)
        model_name: Model identifier matching the bench key (e.g. ``Qwen2.5-14B``
                    for a ``Qwen2.5-14B_tp2`` bench entry). Fuzzy-matched
                    case-insensitively against available keys.
        tp: Tensor parallelism degree used at measurement time (default 2).

    Returns:
        c_comm in seconds (e.g. 3.588e-3 for 3.588 ms)

    Raises:
        FileNotFoundError: bench JSON does not exist
        KeyError: no matching model key in bench
    """
    bench_path = Path(bench_path)
    with open(bench_path) as f:
        data = json.load(f)

    per_model = data.get(f"c_comm_per_model_tp{tp}", data.get("c_comm_per_model_tp2", {}))
    # Fuzzy match: exact, case-insensitive, normalized
    want = f"{model_name}_tp{tp}"
    want_norm = want.replace(".", "").replace("-", "").replace("_", "").lower()
    match = None
    for k in per_model:
        if k == want or k.replace(".", "").replace("-", "").replace("_", "").lower() == want_norm:
            match = k
            break
    if match is None:
        raise KeyError(
            f"No key matching '{want}' in {bench_path}. "
            f"Available: {list(per_model.keys())}. "
            "Re-run measure_nccl_allreduce.py with the target model."
        )

    c_comm = per_model[match].get("c_comm_B32_sec")
    if c_comm is None:
        raise ValueError(
            f"No c_comm_B32_sec for '{match}' in {bench_path}."
        )

    print(f"[Stage 0] c_comm = {c_comm*1e3:.3f} ms ({match} @ B=32, from {bench_path.name})")
    return float(c_comm)


@dataclass
class FitDiagnostics:
    """Diagnostics from a fitting stage."""
    R2: float
    rmse_ms: float
    n_points: int
    params: dict


def load_csv(csv_path: str | Path, engine_level_correction: bool = True) -> list[dict]:
    """Load component sweep CSV into list of dicts.

    Expected columns: model, batch, seq, gamma, mode, mean_ms, median_ms, ...

    When engine_level_correction=True (default), auto-detects cycle-level T_V
    values (vLLM SD instrumentation serializes γ drafter passes with verify on
    the same CUDA stream, producing T_V_cycle ≈ T_T + γ·T_D instead of the
    verify kernel alone). If cycle-level is detected, T_V is adjusted in-place
    via T_V_engine = T_V_cycle − γ·T_D_full so downstream fits receive the
    verify-alone timing the roofline formula expects.

    Pass engine_level_correction=False to skip (e.g., for pre-v53 kernel-level
    sweeps where no correction is needed).
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "model": row["model"],
                "batch": int(row["batch"]),
                "seq": int(row["seq"]),
                "gamma": int(row["gamma"]),
                "mode": row["mode"],
                "mean_ms": float(row["mean_ms"]),
                "median_ms": float(row["median_ms"]),
            })

    if engine_level_correction:
        _apply_engine_level_correction(rows)

    return rows


def _apply_engine_level_correction(rows: list[dict]) -> None:
    """In-place T_V correction when data is cycle-level.

    Auto-detects by computing median((T_V − T_T) / (γ·T_D)) over matched (B, S, γ)
    points. Ratio ≈ 1 ⇒ cycle-level (T_V includes γ drafter iterations),
    ratio ≈ 0 ⇒ kernel-level. Threshold 0.7 is conservative — kernel-level data
    is left untouched.
    """
    drafter = {
        (r["batch"], r["seq"], r["gamma"]): r["median_ms"]
        for r in rows if r["mode"] == "drafter_decode"
    }
    target_ar = {
        (r["batch"], r["seq"]): r["median_ms"]
        for r in rows if r["mode"] == "target_decode" and r["gamma"] == 0
    }

    ratios: list[float] = []
    for r in rows:
        if r["mode"] != "target_verify" or r["batch"] <= 0 or r["gamma"] <= 0:
            continue
        tt = target_ar.get((r["batch"], r["seq"]))
        td = drafter.get((r["batch"], r["seq"], r["gamma"]))
        if tt is None or td is None or td <= 0:
            continue
        gap = r["median_ms"] - tt
        ratio = gap / (r["gamma"] * td)
        if ratio > 0:
            ratios.append(ratio)

    if not ratios:
        print("[load_csv] insufficient matched points for cycle-level detection; skipping correction")
        return

    median_ratio = float(np.median(ratios))

    if median_ratio < 0.7:
        print(f"[load_csv] T_V is kernel-level (median (T_V−T_T)/(γ·T_D) = {median_ratio:.2f}); no correction applied")
        return

    print(f"[load_csv] T_V is cycle-level (median (T_V−T_T)/(γ·T_D) = {median_ratio:.2f}); "
          f"applying T_V_engine = T_V_cycle − γ·T_D_full")

    n_corrected = 0
    for r in rows:
        if r["mode"] != "target_verify" or r["batch"] <= 0 or r["gamma"] <= 0:
            continue
        td = drafter.get((r["batch"], r["seq"], r["gamma"]))
        if td is None:
            continue
        adjustment = r["gamma"] * td
        r["median_ms"] = max(0.1, r["median_ms"] - adjustment)
        r["mean_ms"] = max(0.1, r["mean_ms"] - adjustment)
        n_corrected += 1

    print(f"[load_csv] adjusted {n_corrected} target_verify rows")


def _filter_data(
    rows: list[dict],
    mode: str,
    gamma: Optional[int] = None,
    min_batch: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter CSV rows by mode and gamma, return (B, S, T_ms) arrays."""
    filtered = [
        r for r in rows
        if r["mode"] == mode
        and (gamma is None or r["gamma"] == gamma)
        and r["batch"] >= min_batch
    ]
    if not filtered:
        raise ValueError(f"No data found for mode={mode}, gamma={gamma}")

    B = np.array([r["batch"] for r in filtered], dtype=float)
    S = np.array([r["seq"] for r in filtered], dtype=float)
    T = np.array([r["median_ms"] for r in filtered], dtype=float)
    return B, S, T


def _compute_R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _fit_BW_eff(
    B: np.ndarray, S: np.ndarray, T_ms: np.ndarray,
    mc: ModelConstants,
    F_eff: float,
    gpu: str = "A100",
    c_comm: float = 0.0,
) -> tuple[float, float, float, FitDiagnostics]:
    """Stage 1: Joint fit of (BW_eff, kappa_ratio, c_T) from gamma=0 T_T data.

    Previously BW_eff was pinned from B=1 median under the assumption that
    overhead (c_T, c_comm) was negligible at B=1. That assumption is wrong:
    kernel launch / Python scheduling adds ~0.5-2ms of overhead that is
    comparable to T_T(B=1). Joint fit lets BW_eff, kappa_ratio, c_T compete
    honestly without one absorbing the other's physics.

    c_comm is taken as a FIXED input (pre-measured NCCL overhead), not fitted.

    Joint fit via differential_evolution + L-BFGS-B multi-start on log-space
    MSE, with physically-motivated seed point.

    Returns:
        (BW_eff, kappa_eff, c_T_us, diagnostics)
    """
    T_sec = T_ms * 1e-3
    BW_peak_seed = GPU_SPECS.get(gpu, {}).get("BW_peak", 2.0e12)

    def loss(params):
        BW_eff, kap_ratio, c_T = params
        kappa_eff = mc.kappa_theoretical * kap_ratio
        T_pred = predict_T_T(B, S, mc.W_t, kappa_eff, BW_eff,
                              mc.C_dense, mc.C_attn, F_eff, c_T, c_comm)
        log_err = np.log(T_pred + 1e-12) - np.log(T_sec + 1e-12)
        return float(np.mean(log_err ** 2))

    bounds = [
        (0.4 * BW_peak_seed, 1.05 * BW_peak_seed),
        (0.1, 2.5),
        (0.0, 200.0),
    ]
    x0 = [0.75 * BW_peak_seed, 0.5, 10.0]

    result_de = differential_evolution(loss, bounds, seed=42, maxiter=300, tol=1e-8, workers=1)
    result_lb = minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")
    best = result_lb if result_lb.fun < result_de.fun else result_de

    BW_eff, kap_ratio, c_T_us = best.x
    kappa_eff = mc.kappa_theoretical * kap_ratio

    T_pred_ms = predict_T_T(B, S, mc.W_t, kappa_eff, BW_eff,
                             mc.C_dense, mc.C_attn, F_eff, c_T_us, c_comm) * 1e3
    R2 = _compute_R2(T_ms, T_pred_ms)
    rmse = float(np.sqrt(np.mean((T_ms - T_pred_ms) ** 2)))

    diag = FitDiagnostics(R2=R2, rmse_ms=rmse, n_points=len(T_ms),
                           params={"BW_eff": BW_eff, "kappa_eff_tt": kappa_eff,
                                    "kap_ratio_tt": kap_ratio, "c_T_us": c_T_us})
    return BW_eff, kappa_eff, float(c_T_us), diag


def _fit_S2_shared(
    rows: list[dict],
    gammas: list[int],
    mc: ModelConstants,
    BW_eff: float,
    kappa_eff: float,
    F_eff: float,
    c_T: float,
    c_comm: float = 0.0,
) -> tuple[float, float, float, FitDiagnostics]:
    """Stage 2: Fit shared (η_d, c_D, c_V) from T_D + T_V across all gammas.

    Uses hybrid loss = 0.5·component + 0.5·lv2_primitive.
    All three params are γ-independent.
    BW_eff, κ, c_T are frozen from Stage 1.

    The component loss is log-MSE on T_D and T_V individually.
    The lv2-primitive loss fits T_cycle = γ·T_D + T_V and
    speedup ratio T_T / T_cycle at matched (B, S) points.

    Returns:
        (eta_d, c_D, c_V, diagnostics)
    """
    # Collect per-gamma data dicts
    all_data = []
    for g in gammas:
        try:
            B_td, S_td, T_td = _filter_data(rows, "drafter_decode", g)
            B_tv, S_tv, T_tv = _filter_data(rows, "target_verify", g)
        except ValueError:
            continue

        T_td_sec = T_td * 1e-3
        T_tv_sec = T_tv * 1e-3

        # Build T_T lookup for lv2-primitive at matched points
        try:
            B_tt_g, S_tt_g, T_tt_g = _filter_data(rows, "target_decode", gamma=0)
        except ValueError:
            B_tt_g, S_tt_g, T_tt_g = np.array([]), np.array([]), np.array([])

        tt_lookup = {(int(b), int(s)): t for b, s, t in zip(B_tt_g, S_tt_g, T_tt_g)}

        td_pairs = set(zip(B_td.astype(int), S_td.astype(int)))
        tv_pairs = set(zip(B_tv.astype(int), S_tv.astype(int)))
        common = sorted(td_pairs & tv_pairs & set(tt_lookup.keys()))

        if common:
            sp_B = np.array([b for b, s in common], dtype=float)
            sp_S = np.array([s for b, s in common], dtype=float)
            sp_TT = np.array([tt_lookup[(b, s)] for b, s in common]) * 1e-3
            sp_TD = np.array([T_td[np.where((B_td == b) & (S_td == s))[0][0]]
                               for b, s in common]) * 1e-3
            sp_TV = np.array([T_tv[np.where((B_tv == b) & (S_tv == s))[0][0]]
                               for b, s in common]) * 1e-3
            sp_cyc = g * sp_TD + sp_TV
        else:
            sp_B = sp_S = sp_TT = sp_TD = sp_TV = sp_cyc = np.array([])

        all_data.append({
            "gamma": g,
            "B_td": B_td, "S_td": S_td, "T_td_sec": T_td_sec,
            "B_tv": B_tv, "S_tv": S_tv, "T_tv_sec": T_tv_sec,
            "sp_B": sp_B, "sp_S": sp_S,
            "sp_TT": sp_TT, "sp_cyc": sp_cyc,
        })

    if not all_data:
        raise ValueError("No drafter/verify data found for any gamma")

    def loss(p):
        eta_d, c_D, c_V = p
        total = 0.0
        for d in all_data:
            g = d["gamma"]
            # Component loss (log-MSE on T_D and T_V individually)
            Td_a = predict_T_D(d["B_td"], d["S_td"], mc.W_d, eta_d, kappa_eff, BW_eff,
                                mc.C_dense, mc.C_attn, F_eff, c_D, c_comm)
            Tv_a = predict_T_V(d["B_tv"], d["S_tv"], g, mc.W_t, kappa_eff, BW_eff,
                                mc.C_dense, mc.C_attn, F_eff, c_V, c_comm)
            ld = np.mean((np.log(Td_a + 1e-12) - np.log(d["T_td_sec"] + 1e-12)) ** 2)
            lv = np.mean((np.log(Tv_a + 1e-12) - np.log(d["T_tv_sec"] + 1e-12)) ** 2)
            comp_loss = ld + lv

            # Lv2-primitive loss (T_cycle + speedup ratio)
            if len(d["sp_B"]) > 0:
                Tt = predict_T_T(d["sp_B"], d["sp_S"], mc.W_t, kappa_eff, BW_eff,
                                  mc.C_dense, mc.C_attn, F_eff, c_T, c_comm)
                Td = predict_T_D(d["sp_B"], d["sp_S"], mc.W_d, eta_d, kappa_eff, BW_eff,
                                  mc.C_dense, mc.C_attn, F_eff, c_D, c_comm)
                Tv = predict_T_V(d["sp_B"], d["sp_S"], g, mc.W_t, kappa_eff, BW_eff,
                                  mc.C_dense, mc.C_attn, F_eff, c_V, c_comm)
                cyc_p = g * Td + Tv
                lc = np.mean((np.log(cyc_p + 1e-12) - np.log(d["sp_cyc"] + 1e-12)) ** 2)
                ls = np.mean((Tt / cyc_p - d["sp_TT"] / d["sp_cyc"]) ** 2)
                lv2_loss = lc + ls
            else:
                lv2_loss = comp_loss  # fallback when no matched T_T

            total += 0.5 * comp_loss + 0.5 * lv2_loss
        return float(total / len(all_data))

    bounds = [
        (1.0, 4.0),    # eta_d
        (0.0, 200.0),  # c_D (μs)
        (0.0, 500.0),  # c_V (μs)
    ]

    result_de = differential_evolution(loss, bounds, seed=42, maxiter=500, tol=1e-10, workers=1)
    result_lb = minimize(loss, x0=[1.4, 30.0, 100.0], bounds=bounds, method="L-BFGS-B")
    best = result_lb if result_lb.fun < result_de.fun else result_de

    eta_d, c_D, c_V = best.x

    # Diagnostics: averaged R² across gammas
    R2_vals, rmse_vals, n_pts = [], [], 0
    for d in all_data:
        g = d["gamma"]
        T_D_pred_ms = predict_T_D(d["B_td"], d["S_td"], mc.W_d, eta_d, kappa_eff, BW_eff,
                                   mc.C_dense, mc.C_attn, F_eff, c_D, c_comm) * 1e3
        T_V_pred_ms = predict_T_V(d["B_tv"], d["S_tv"], g, mc.W_t, kappa_eff, BW_eff,
                                   mc.C_dense, mc.C_attn, F_eff, c_V, c_comm) * 1e3
        T_td_ms = d["T_td_sec"] * 1e3
        T_tv_ms = d["T_tv_sec"] * 1e3
        R2_d = _compute_R2(T_td_ms, T_D_pred_ms)
        R2_v = _compute_R2(T_tv_ms, T_V_pred_ms)
        R2_vals.append((R2_d + R2_v) / 2)
        rmse_d = float(np.sqrt(np.mean((T_td_ms - T_D_pred_ms) ** 2)))
        rmse_v = float(np.sqrt(np.mean((T_tv_ms - T_V_pred_ms) ** 2)))
        rmse_vals.append((rmse_d + rmse_v) / 2)
        n_pts += len(d["B_td"]) + len(d["B_tv"])

    R2_mean = float(np.mean(R2_vals))
    rmse_mean = float(np.mean(rmse_vals))

    diag = FitDiagnostics(
        R2=R2_mean,
        rmse_ms=rmse_mean,
        n_points=n_pts,
        params={
            "eta_d": round(float(eta_d), 3),
            "c_D": round(float(c_D), 1),
            "c_V": round(float(c_V), 1),
            "loss": round(float(best.fun), 6),
        },
    )
    return float(eta_d), float(c_D), float(c_V), diag


def calibrate(
    csv_path: str | Path,
    model_name: str,
    tp: int = 1,
    gpu: str = "A100",
    quant_ratio: float = 0.25,
    gammas: Optional[list[int]] = None,
    F_eff: float = 200e12,
    F_eff_bench_path: Optional[str | Path] = None,
    engine_level_correction: bool = True,
    c_comm: float = 0.0,
    c_comm_bench_path: Optional[str | Path] = None,
) -> SDToggleConfig:
    """Run full calibration pipeline on existing CSV data.

    3-stage pipeline — all 6 fitted params are γ-independent:

      Stage 0: (Optional) Load F_eff from GEMM micro-benchmark.
      Stage 1: Fit (BW_eff, κ, c_T) from T_T (target_decode, gamma=0) data.
      Stage 2: Fit (η_d, c_D, c_V) shared across ALL gammas simultaneously
               using hybrid loss = 0.5·component_logMSE + 0.5·lv2_primitive.

    β=0 fixed (tried, marginal gain, dropped).
    No per-gamma loop, no averaging — all params are truly γ-independent.

    Args:
        csv_path: Path to finegrain component sweep CSV
        model_name: Model identifier (must be in constants.MODEL_CONSTANTS)
        tp: Tensor parallelism degree
        gpu: GPU name for BW_peak / F_peak metadata and fallback BW_eff
        quant_ratio: Drafter quantization ratio (fraction of weights quantized)
        gammas: Gamma values to calibrate (auto-detected from CSV if None)
        F_eff: Fixed effective compute throughput in FLOPS (default: 200e12 = 200T).
               Ignored when F_eff_bench_path is provided.
        F_eff_bench_path: Path to GEMM bench JSON from measure_gemm_effective_tflops.py.
               When provided, F_eff is loaded from the bench (Stage 0) and the
               F_eff argument is ignored.
        engine_level_correction: Auto-detect and correct cycle-level T_V
        c_comm: Pre-measured NCCL overhead per forward pass (seconds).
               Ignored when c_comm_bench_path is provided.
        c_comm_bench_path: Path to NCCL bench JSON from measure_nccl_allreduce.py.
               When provided, c_comm is loaded via load_c_comm_from_bench (Stage 0)
               using model_name for lookup.

    Returns:
        Calibrated SDToggleConfig ready for serialization and deployment
    """
    csv_path = Path(csv_path)

    # Stage 0: Load F_eff from GEMM bench if provided
    F_eff_source = "default"
    if F_eff_bench_path is not None:
        F_eff = load_F_eff_from_bench(F_eff_bench_path)
        F_eff_source = str(F_eff_bench_path)

    # Stage 0: Load c_comm from NCCL bench if provided (tp>1 only meaningful)
    c_comm_source = "default"
    if c_comm_bench_path is not None:
        c_comm = load_c_comm_from_bench(c_comm_bench_path, model_name, tp=tp)
        c_comm_source = str(c_comm_bench_path)

    rows = load_csv(csv_path, engine_level_correction=engine_level_correction)

    mc = compute_constants(model_name, tp=tp, quant_ratio=quant_ratio)

    # Auto-detect gammas from CSV
    if gammas is None:
        gammas = sorted(set(
            r["gamma"] for r in rows
            if r["gamma"] > 0 and r["mode"] in ("drafter_decode", "target_verify")
        ))
    if not gammas:
        raise ValueError("No gamma > 0 data found in CSV")

    print(f"Calibrating {model_name} (tp={tp}) from {csv_path.name}")
    print(f"  Gammas: {gammas}")

    # Stage 1: (BW_eff, κ, c_T) from T_T (target decode, gamma=0)
    B_tt, S_tt, T_tt = _filter_data(rows, "target_decode", gamma=0)
    BW_eff, kappa_from_tt, c_T_stage1, bw_diag = _fit_BW_eff(
        B_tt, S_tt, T_tt, mc, F_eff, gpu=gpu, c_comm=c_comm,
    )
    print(f"  Stage 1: BW_eff = {BW_eff/1e12:.3f} TB/s, "
          f"kappa = {kappa_from_tt:.0f} ({kappa_from_tt/mc.kappa_theoretical:.2f}x), "
          f"c_T = {c_T_stage1:.1f} μs, R2={bw_diag.R2:.3f}")

    # Stage 2: Shared (η_d, c_D, c_V) across all gammas — hybrid loss
    eta_d, c_D, c_V, s2_diag = _fit_S2_shared(
        rows, gammas, mc, BW_eff, kappa_from_tt, F_eff,
        c_T=c_T_stage1, c_comm=c_comm,
    )
    print(f"  Stage 2: eta_d={eta_d:.3f}, c_D={c_D:.1f}, c_V={c_V:.1f}, "
          f"R2={s2_diag.R2:.3f}, loss={s2_diag.params['loss']:.6f}")

    gpu_spec = GPU_SPECS.get(gpu, {"BW_peak": 0.0, "F_peak": 0.0})

    config = SDToggleConfig(
        hardware=HardwareConfig(
            gpu=gpu,
            tp=tp,
            BW_eff=BW_eff,
            BW_peak=gpu_spec["BW_peak"],
            F_peak=gpu_spec["F_peak"],
            c_comm=c_comm,
        ),
        model=ModelConfig(
            name=mc.name,
            W_t=mc.W_t,
            W_d=mc.W_d,
            C_dense=mc.C_dense,
            C_attn=mc.C_attn,
            kappa_theoretical=mc.kappa_theoretical,
            rho=mc.rho,
            gqa=mc.gqa,
        ),
        calibration=CalibrationConfig(
            eta_d=round(eta_d, 3),
            kappa_eff=round(kappa_from_tt),
            F_eff=F_eff,
            c_T=round(c_T_stage1, 1),
            c_D=round(c_D, 1),
            c_V=round(c_V, 1),
            beta=0.0,
            per_gamma={},
            per_gamma_full=None,
        ),
        metadata={
            "created": str(np.datetime64("today")),
            "csv_source": csv_path.name,
            "F_eff_tflops": round(F_eff / 1e12, 1),
            "F_eff_source": F_eff_source,
            "c_comm_ms": round(c_comm * 1e3, 4),
            "c_comm_source": c_comm_source,
            "phase": 1,
            "fit_loss": "hybrid_0.5comp_0.5lv2",
            "gammas": gammas,
            "s2_diagnostics": {
                "R2": s2_diag.R2,
                "rmse_ms": s2_diag.rmse_ms,
                "n_points": s2_diag.n_points,
                **s2_diag.params,
            },
            "bw_diagnostics": {
                "BW_eff": BW_eff,
                "R2": bw_diag.R2,
                "n_points": bw_diag.n_points,
            },
        },
    )

    return config
