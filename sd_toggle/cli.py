"""CLI entry points for sd_toggle package.

Usage:
    python -m sd_toggle sweep --model X --gpu 0 [--gammas 3,7,11,15] [--batches 1,2,4] [--seq-lens 512,1024] [--output sweep.csv]
    python -m sd_toggle calibrate --csv PATH --model NAME [--tp 1] [--gpu A100]
    python -m sd_toggle calibrate --model NAME --gpu-id 0  # auto-sweep then fit
    python -m sd_toggle plot --config PATH [--csv PATH] [--output DIR]
    python -m sd_toggle predict --config PATH --batch B --seq S --gamma G --L-accept L
    python -m sd_toggle info --config PATH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run component sweep measurement."""
    gammas = [int(g) for g in args.gammas.split(",")] if args.gammas else None
    batches = [int(b) for b in args.batches.split(",")] if args.batches else None
    seq_lens = [int(s) for s in args.seq_lens.split(",")] if args.seq_lens else None

    if args.gpus:
        # Multi-GPU parallel sweep
        from .sweep import run_sweep_multi_gpu
        gpu_list = [int(g) for g in args.gpus.split(",")]
        output_path = run_sweep_multi_gpu(
            model=args.model,
            gpus=gpu_list,
            gammas=gammas,
            batches=batches,
            seq_lens=seq_lens,
            tp=args.tp,
            output=args.output,
        )
    else:
        # Single-GPU sweep (backward compatible)
        from .sweep import run_sweep
        output_path = run_sweep(
            model=args.model,
            gpu=args.gpu,
            gammas=gammas,
            batches=batches,
            seq_lens=seq_lens,
            tp=args.tp,
            output=args.output,
        )
    print(f"\nSweep complete. Results saved to: {output_path}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run calibration pipeline."""
    from .fit import calibrate
    from .config import save_config
    from .constants import compute_constants
    import json

    gammas = None
    if args.gammas:
        gammas = [int(g) for g in args.gammas.split(",")]

    # Infer TP from --gpu-id (e.g., "0" → tp=1, "0,1" → tp=2)
    tp = args.tp
    gpu_ids = None
    if args.gpu_id is not None:
        gpu_ids = [int(g) for g in str(args.gpu_id).split(",")]
        if tp is None:
            tp = len(gpu_ids)
    if tp is None:
        tp = 1

    csv_path = args.csv
    if csv_path is None and gpu_ids is not None:
        from .sweep import run_sweep
        primary_gpu = gpu_ids[0]
        print(f"No --csv provided; running auto-sweep on GPU {gpu_ids} (tp={tp})...")
        csv_path = str(run_sweep(
            model=args.model,
            gpu=primary_gpu,
            gammas=gammas,
            tp=tp,
        ))
        print(f"Auto-sweep complete. Using: {csv_path}")

    if csv_path is None:
        print("Error: must provide --csv or --gpu-id", file=sys.stderr)
        sys.exit(1)

    config = calibrate(
        csv_path=csv_path,
        model_name=args.model,
        tp=tp,
        gpu=args.gpu,
        gammas=gammas,
        F_eff=args.F_eff,
        F_eff_bench_path=getattr(args, 'F_eff_bench', None),
        c_comm=getattr(args, 'c_comm', 0.0),
        c_comm_bench_path=getattr(args, 'c_comm_bench', None),
    )

    output_dir = Path(args.output)
    if output_dir.is_dir() or args.output.endswith("/"):
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)

    model_short = config.model.name.replace(".", "").replace("-", "").lower()
    gpu_short = args.gpu.lower().split("-")[0]

    # Save calibrated config
    config_path = output_dir / f"{gpu_short}_tp{tp}_{model_short}.json"
    if not Path(args.output).is_dir() and not args.output.endswith("/"):
        config_path = Path(args.output)
    save_config(config, config_path)
    print(f"\nConfig saved: {config_path}")

    # Also save model constants
    mc = compute_constants(args.model, tp=tp)
    mc_data = {
        "model": mc.name, "tp": mc.tp, "quant_ratio": mc.quant_ratio,
        "D": mc.D, "D_ff": mc.D_ff, "L": mc.L,
        "n_heads": mc.n_heads, "n_kv": mc.n_kv, "d_h": mc.d_h,
        "V": mc.V, "gqa": mc.gqa,
        "W_t": mc.W_t, "W_t_GB": mc.W_t / 1e9,
        "W_d": mc.W_d, "W_d_GB": mc.W_d / 1e9,
        "rho": mc.rho,
        "kappa_theoretical": mc.kappa_theoretical,
        "C_dense": mc.C_dense, "C_dense_GFLOPS": mc.C_dense / 1e9,
        "C_attn": mc.C_attn, "C_attn_MFLOPS": mc.C_attn / 1e6,
    }
    mc_path = output_dir / f"model_constants_{model_short}_tp{tp}.json"
    with open(mc_path, "w") as f:
        json.dump(mc_data, f, indent=2)
    print(f"Model constants saved: {mc_path}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate boundary plots."""
    from .config import load_config
    from .plot import plot_boundary_with_empirical, plot_Laccept_sweep
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    config = load_config(args.config)
    gammas = sorted(config.calibration.per_gamma.keys())
    if not gammas:
        gammas = config.metadata.get("gammas", [])
    if not gammas:
        print("Error: no gamma values found in config. Provide --gammas.", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    L_accepts = [float(x) for x in args.L_accepts.split(",")]

    for gamma in gammas:
        for L in L_accepts:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            if args.csv:
                plot_boundary_with_empirical(
                    config, gamma, L, args.csv, ax=ax,
                    pow2_filter=args.pow2,
                )
            else:
                from .plot import plot_boundary
                plot_boundary(config, gamma, L, ax=ax)

            stem = f"{config.model.name}_gamma{gamma}_L{L:.1f}"
            fig.savefig(output_dir / f"{stem}.png", dpi=300)
            fig.savefig(output_dir / f"{stem}.pdf")
            plt.close(fig)
            print(f"Saved: {output_dir / stem}.png/.pdf")

        # L_accept sweep per gamma
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_Laccept_sweep(config, gamma, L_accepts, ax=ax)
        stem = f"{config.model.name}_gamma{gamma}_Lsweep"
        fig.savefig(output_dir / f"{stem}.png", dpi=300)
        plt.close(fig)
        print(f"Saved: {output_dir / stem}.png")


def cmd_predict(args: argparse.Namespace) -> None:
    """Single-point prediction."""
    from .config import load_config
    from .predict import predict_decision

    config = load_config(args.config)

    L = args.L_accept
    if L is None:
        L = float(args.gamma)
        print(f"No --L-accept given; defaulting to L=γ={L} (trained-policy regime)", file=sys.stderr)

    result = predict_decision(config, args.batch, args.seq, args.gamma, L)

    print(f"\nPrediction for B={args.batch}, S={args.seq}, γ={args.gamma}, L={L:.2f}:")
    print(f"  SD enabled: {result['sd_on']}")
    print(f"  Speedup:    {result['speedup']:.3f}")
    print(f"  r (T_D/T_T): {result['r']:.3f}")
    print(f"  v (T_V/T_T): {result['v']:.3f}")
    print(f"  T_T regime:  {result['regime_T_T']}")
    print(f"  T_V regime:  {result['regime_T_V']}")


def cmd_info(args: argparse.Namespace) -> None:
    """Print config summary."""
    from .config import load_config
    config = load_config(args.config)

    print(f"SD Toggle Config: {args.config}")
    print(f"  Hardware: {config.hardware.gpu} (TP={config.hardware.tp})")
    print(f"  BW_eff: {config.hardware.BW_eff/1e12:.3f} TB/s")
    print(f"  Model: {config.model.name}")
    print(f"  W_t: {config.model.W_t/1e9:.2f} GB, W_d: {config.model.W_d/1e9:.2f} GB")
    print(f"  eta_d: {config.calibration.eta_d:.3f}")
    print(f"  kappa_eff: {config.calibration.kappa_eff:.0f}")
    print(f"  F_eff: {config.calibration.F_eff/1e12:.0f} TFLOPS")
    if config.calibration.per_gamma:
        print(f"  Gammas: {sorted(config.calibration.per_gamma.keys())}")
        for g, cal in sorted(config.calibration.per_gamma.items()):
            print(f"    γ={g}: c_D={cal.c_D}, c_V={cal.c_V}, c_T={cal.c_T}, R²={cal.R2}")
    else:
        gammas = config.metadata.get("gammas", [])
        print(f"  Gammas: {gammas} (shared params)")
        print(f"  c_T={config.calibration.c_T}, c_D={config.calibration.c_D}, c_V={config.calibration.c_V}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate calibrated config against sweep data."""
    import numpy as np
    from .config import load_config
    from .fit import load_csv, _filter_data, _compute_R2
    from .roofline import predict_T_T, predict_T_D, predict_T_V

    config = load_config(args.config)
    rows = load_csv(args.csv)
    cal = config.calibration
    hw = config.hardware
    md = config.model

    # Resolve L_accept: fixed if given, otherwise per-γ default L=γ (trained regime)
    if args.L_accept is not None:
        L_accept_fixed = args.L_accept
        L_label = f"L̄={L_accept_fixed:.2f}"
    else:
        L_accept_fixed = None  # signal: L = gamma per iteration
        L_label = "L̄=γ (per-γ default)"

    gammas = sorted(set(
        r["gamma"] for r in rows
        if r["gamma"] > 0 and r["mode"] in ("drafter_decode", "target_verify")
    ))

    # Get overhead params (shared or per-gamma)
    gc = config.get_gamma_params(gammas[0])
    c_T, c_D, c_V = gc.c_T, gc.c_D, gc.c_V

    print(f"Validating: {args.config}")
    print(f"  Model: {md.name}, GPU: {hw.gpu} (TP={hw.tp})")
    print(f"  Params: BW={hw.BW_eff/1e12:.3f} TB/s, κ={cal.kappa_eff:.0f}, "
          f"η_d={cal.eta_d:.3f}, c_T={c_T:.1f}, c_D={c_D:.1f}, c_V={c_V:.1f}")
    print(f"  F_eff={cal.F_eff/1e12:.1f}T, {L_label}, gammas={gammas}")

    # ── T_T ──
    B_tt, S_tt, T_tt = _filter_data(rows, "target_decode", gamma=0)
    T_tt_pred = predict_T_T(B_tt, S_tt, md.W_t, cal.kappa_eff, hw.BW_eff,
                             md.C_dense, md.C_attn, cal.F_eff, c_T, hw.c_comm) * 1e3
    R2_tt = _compute_R2(T_tt, T_tt_pred)
    mape_tt = np.mean(np.abs(T_tt - T_tt_pred) / T_tt) * 100
    bias_tt = np.mean((T_tt_pred - T_tt) / T_tt) * 100
    print(f"\n{'='*80}")
    print(f"  T_T:  R²={R2_tt:.3f}  MAPE={mape_tt:.1f}%  bias={bias_tt:+.1f}%  n={len(T_tt)}")

    tt_lookup = {(int(b), int(s)): t for b, s, t in zip(B_tt, S_tt, T_tt)}

    # ── Per-gamma ──
    header = (f"  {'γ':>3} │ {'T_D R²':>6} {'MAPE':>5} {'bias':>6} │ "
              f"{'T_V R²':>6} {'MAPE':>5} {'bias':>6} │ "
              f"{'r_bias':>7} {'r_MAE':>6} │ {'v_bias':>7} {'v_MAE':>6} │ "
              f"{'cyc%':>5} │ {'spd_b':>6} {'MAE':>5} │ {'sign':>10}")
    print(f"\n{header}")
    print(f"  {'─'*100}")

    all_sign_ok = all_sign_tot = 0
    for gamma in gammas:
        gc_g = config.get_gamma_params(gamma)
        L = L_accept_fixed if L_accept_fixed is not None else float(gamma)

        B_td, S_td, T_td = _filter_data(rows, "drafter_decode", gamma)
        B_tv, S_tv, T_tv = _filter_data(rows, "target_verify", gamma)

        Td_pred = predict_T_D(B_td, S_td, md.W_d, cal.eta_d, cal.kappa_eff, hw.BW_eff,
                               md.C_dense, md.C_attn, cal.F_eff, gc_g.c_D, hw.c_comm) * 1e3
        Tv_pred = predict_T_V(B_tv, S_tv, gamma, md.W_t, cal.kappa_eff, hw.BW_eff,
                               md.C_dense, md.C_attn, cal.F_eff, gc_g.c_V, hw.c_comm,
                               beta=cal.beta) * 1e3

        R2_d = _compute_R2(T_td, Td_pred)
        R2_v = _compute_R2(T_tv, Tv_pred)
        mape_d = np.mean(np.abs(Td_pred - T_td) / T_td) * 100
        mape_v = np.mean(np.abs(Tv_pred - T_tv) / T_tv) * 100
        bias_d = np.mean((Td_pred - T_td) / T_td) * 100
        bias_v = np.mean((Tv_pred - T_tv) / T_tv) * 100

        # Matched points for ratios
        td_lk = {(int(b), int(s)): t for b, s, t in zip(B_td, S_td, T_td)}
        tv_lk = {(int(b), int(s)): t for b, s, t in zip(B_tv, S_tv, T_tv)}
        common = sorted(set(tt_lookup) & set(td_lk) & set(tv_lk))

        r_errs, v_errs, cyc_errs, spd_errs = [], [], [], []
        sign_ok = sign_tot = 0
        for b, s in common:
            ttm, tdm, tvm = tt_lookup[(b, s)], td_lk[(b, s)], tv_lk[(b, s)]
            ttp = predict_T_T(b, s, md.W_t, cal.kappa_eff, hw.BW_eff,
                               md.C_dense, md.C_attn, cal.F_eff, gc_g.c_T, hw.c_comm) * 1e3
            tdp = predict_T_D(b, s, md.W_d, cal.eta_d, cal.kappa_eff, hw.BW_eff,
                               md.C_dense, md.C_attn, cal.F_eff, gc_g.c_D, hw.c_comm) * 1e3
            tvp = predict_T_V(b, s, gamma, md.W_t, cal.kappa_eff, hw.BW_eff,
                               md.C_dense, md.C_attn, cal.F_eff, gc_g.c_V, hw.c_comm,
                               beta=cal.beta) * 1e3

            r_errs.append(tdp / ttp - tdm / ttm)
            v_errs.append(tvp / ttp - tvm / ttm)
            cyc_m = gamma * tdm + tvm
            cyc_p = gamma * tdp + tvp
            cyc_errs.append((cyc_p - cyc_m) / cyc_m * 100)
            spd_m = L * ttm / cyc_m
            spd_p = L * ttp / cyc_p
            spd_errs.append(spd_p - spd_m)
            sign_tot += 1
            if (spd_m > 1) == (spd_p > 1):
                sign_ok += 1

        r_errs = np.array(r_errs)
        v_errs = np.array(v_errs)
        cyc_errs = np.array(cyc_errs)
        spd_errs = np.array(spd_errs)
        all_sign_ok += sign_ok
        all_sign_tot += sign_tot

        print(f"  {gamma:3d} │ {R2_d:.3f} {mape_d:5.1f} {bias_d:+5.1f}% │ "
              f"{R2_v:.3f} {mape_v:5.1f} {bias_v:+5.1f}% │ "
              f"{np.mean(r_errs):+7.4f} {np.mean(np.abs(r_errs)):6.4f} │ "
              f"{np.mean(v_errs):+7.4f} {np.mean(np.abs(v_errs)):6.4f} │ "
              f"{np.mean(np.abs(cyc_errs)):5.1f} │ "
              f"{np.mean(spd_errs):+6.4f} {np.mean(np.abs(spd_errs)):5.4f} │ "
              f"{sign_ok:4d}/{sign_tot}={sign_ok/sign_tot*100:.0f}%")

    sign_pct = all_sign_ok / all_sign_tot * 100 if all_sign_tot else 0
    print(f"  {'─'*100}")
    print(f"  ALL │ T_T MAPE={mape_tt:.1f}%  │  Toggle sign = {all_sign_ok}/{all_sign_tot} = {sign_pct:.0f}%")

    # ── Pass/Fail ──
    print(f"\n{'='*80}")
    bw_frac = hw.BW_eff / hw.BW_peak if hw.BW_peak else 0.0
    checks = [
        ("T_T MAPE < 10%", mape_tt < 10),
        ("Sign accuracy ≥ 90%", sign_pct >= 90),
        (
            f"BW_eff = {hw.BW_eff/1e12:.2f} TB/s ({bw_frac*100:.0f}% of peak {hw.BW_peak/1e12:.2f})  [expect ≥ 40%]",
            0.40 <= bw_frac <= 1.05,
        ),
        ("η_d in [1.0, 3.0]", 1.0 <= cal.eta_d <= 3.0),
    ]
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  {'VALIDATION PASSED' if all_pass else 'VALIDATION FAILED — check diagnostics above'}")


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sd_toggle",
        description="SD Toggle: Roofline-aware speculative decoding toggle calibration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sweep
    p_sw = subparsers.add_parser("sweep", help="Run vLLM component sweep measurement")
    p_sw.add_argument("--model", required=True, help="Model ID (e.g. Qwen/Qwen2.5-7B)")
    p_sw.add_argument("--gpu", type=int, default=0, help="GPU index (single-GPU mode)")
    p_sw.add_argument("--gpus", default=None,
                       help="Comma-separated GPU indices for multi-GPU parallel sweep. "
                            "For TP>1, list PRIMARY indices only — each primary p "
                            "expands to pair (p, p+1, …, p+tp-1). "
                            "Examples: TP=1 '--gpus 0,1,2,3', TP=2 '--gpus 0,2,4,6 --tp 2'. "
                            "Overrides --gpu.")
    p_sw.add_argument("--gammas", default=None, help="Comma-separated gamma values (default: 3,7,11,15)")
    p_sw.add_argument("--batches", default=None, help="Comma-separated batch sizes")
    p_sw.add_argument("--seq-lens", default=None, help="Comma-separated sequence lengths")
    p_sw.add_argument("--tp", type=int, default=1, help="Tensor parallelism (default: 1)")
    p_sw.add_argument("--output", default=None, help="Output CSV path")
    p_sw.set_defaults(func=cmd_sweep)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Run calibration from CSV data")
    p_cal.add_argument("--csv", default=None, help="Path to component sweep CSV")
    p_cal.add_argument("--model", required=True, help="Model name (e.g. Qwen2.5-7B)")
    p_cal.add_argument("--tp", type=int, default=None,
                       help="Tensor parallelism (auto-detected from --gpu-id count if omitted)")
    p_cal.add_argument("--gpu", default="A100", help="GPU name (e.g. A100)")
    p_cal.add_argument("--gpu-id", default=None,
                       help="GPU index(es) for auto-sweep, comma-separated (e.g. '0' for tp=1, '0,1' for tp=2)")
    p_cal.add_argument("--gammas", default=None, help="Comma-separated gammas")
    p_cal.add_argument("--F-eff", type=float, default=200e12, help="Fixed F_eff (FLOPS)")
    p_cal.add_argument("--F-eff-bench", default=None,
                       help="Path to GEMM bench JSON for F_eff (overrides --F-eff)")
    p_cal.add_argument("--c-comm", type=float, default=0.0,
                       help="Pre-measured NCCL overhead per forward pass (seconds, tp>1 only)")
    p_cal.add_argument("--c-comm-bench", default=None,
                       help="Path to NCCL bench JSON for c_comm (overrides --c-comm, looks up by model)")
    p_cal.add_argument("--output", default="sd_toggle/configs/", help="Output path or dir")
    p_cal.set_defaults(func=cmd_calibrate)

    # plot
    p_plot = subparsers.add_parser("plot", help="Generate boundary plots")
    p_plot.add_argument("--config", required=True, help="Config JSON path")
    p_plot.add_argument("--csv", default=None, help="CSV for empirical overlay")
    p_plot.add_argument("--output", default="results/sd_toggle_plots/", help="Output dir")
    p_plot.add_argument("--L-accepts", default="3.0,7.0,11.0,15.0", help="L_accept values (default: γ values for the calibrated set)")
    p_plot.add_argument("--pow2", action="store_true", default=True, help="Power-of-2 batch filter")
    p_plot.add_argument("--no-pow2", dest="pow2", action="store_false")
    p_plot.set_defaults(func=cmd_plot)

    # predict
    p_pred = subparsers.add_parser("predict", help="Single-point toggle prediction")
    p_pred.add_argument("--config", required=True, help="Config JSON path")
    p_pred.add_argument("--batch", type=int, required=True)
    p_pred.add_argument("--seq", type=int, required=True)
    p_pred.add_argument("--gamma", type=int, required=True)
    p_pred.add_argument("--L-accept", type=float, default=None, help="Expected accepted length")
    p_pred.set_defaults(func=cmd_predict)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate config against sweep data")
    p_val.add_argument("--config", required=True, help="Config JSON path")
    p_val.add_argument("--csv", required=True, help="Sweep CSV path")
    p_val.add_argument("--L-accept", type=float, default=None,
                       help="Expected accepted length (e.g. 3.7). Used for all gammas.")
    p_val.set_defaults(func=cmd_validate)

    # info
    p_info = subparsers.add_parser("info", help="Print config summary")
    p_info.add_argument("--config", required=True, help="Config JSON path")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
