"""Measure sustained fp16 tensor-core TFLOPS at decode-representative GEMM shapes.

For each (model, tp) triple, benchmarks the four matrix multiplications that
dominate a single transformer decode forward pass:

  attn_qkv  : (B, D) @ (D, (n_heads + 2*n_kv)*d_h // tp)
  attn_out  : (B, n_heads*d_h // tp) @ (n_heads*d_h // tp, D)
  ffn_gate_up: (B, D) @ (D, 2*D_ff // tp)
  ffn_down  : (B, D_ff // tp) @ (D_ff // tp, D)

Usage:
    python scripts/profiling/measure_gemm_effective_tflops.py \
        --models "Qwen2.5-7B:1,Llama-3.1-8B-Instruct:1,Qwen2.5-14B:2" \
        --gpu 0
    # Output defaults to sd_toggle/configs/F_eff_bench_${gpu_short}.json
    # where gpu_short is auto-derived from torch.cuda.get_device_name()
    # (e.g. "NVIDIA A100-SXM4-80GB" → "a100").
    # Pass --output to override.
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sd_toggle.constants import compute_constants


BATCH_SIZES = [1, 4, 16, 32, 64, 128, 256, 512, 1024]
WARMUP = 10
TIMED = 50
# Batch sizes considered representative of the toggle transition regime
TRANSITION_BATCH_SIZES = {16, 32, 64}
# Batch sizes in the compute-bound saturation regime — used to extract F_eff
SATURATION_BATCH_SIZES = {256, 512, 1024}


def tflops_for_matmul(M: int, N: int, K: int, time_s: float) -> float:
    """TFLOPS = 2*M*N*K / time_s / 1e12  (fp16 matmul counts 2 ops per FMA)."""
    return 2.0 * M * N * K / time_s / 1e12


def benchmark_matmul(
    M: int, N: int, K: int, device: torch.device, warmup: int, timed: int
) -> float:
    """Return median latency in seconds for fp16 (M,K)@(K,N) on device."""
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)

    for _ in range(warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        _ = torch.matmul(A, B)
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end) * 1e-3)

    return statistics.median(times)


def measure_model(
    model_name: str, tp: int, device: torch.device
) -> dict[str, Any]:
    mc = compute_constants(model_name, tp=tp)

    D = mc.D
    D_ff = mc.D_ff
    n_heads = mc.n_heads
    n_kv = mc.n_kv
    d_h = mc.d_h

    shapes = {
        "attn_qkv": (D, (n_heads + 2 * n_kv) * d_h // tp),
        "attn_out": (n_heads * d_h // tp, D),
        "ffn_gate_up": (D, 2 * D_ff // tp),
        "ffn_down": (D_ff // tp, D),
    }

    # FLOPs weight per shape = N * K (proportional to per-token compute).
    # Used to compute FLOPs-weighted mean throughput: shapes that dominate
    # C_dense (e.g. ffn_gate_up) get proportionally more weight.
    flops_weight: dict[str, int] = {name: N * K for name, (K, N) in shapes.items()}
    total_weight = sum(flops_weight.values())

    per_shape_B: dict[str, float] = {}
    all_tflops: list[float] = []
    transition_tflops: list[float] = []
    saturation_tflops: list[float] = []
    # Per-batch saturation data for FLOPs-weighted computation
    sat_per_batch: dict[int, dict[str, float]] = {}

    for B in BATCH_SIZES:
        for shape_name, (K, N) in shapes.items():
            M = B
            time_s = benchmark_matmul(M, N, K, device, WARMUP, TIMED)
            tf = tflops_for_matmul(M, N, K, time_s)
            key = f"{shape_name}_B{B}"
            per_shape_B[key] = round(tf, 3)
            all_tflops.append(tf)
            if B in TRANSITION_BATCH_SIZES:
                transition_tflops.append(tf)
            if B in SATURATION_BATCH_SIZES:
                saturation_tflops.append(tf)
                sat_per_batch.setdefault(B, {})[shape_name] = tf
            print(f"  {shape_name:15s} B={B:3d}: {tf:6.1f} TFLOPS  ({time_s*1e3:.3f} ms)")

    # F_eff_saturated (unweighted): median throughput in compute-bound regime
    F_eff_saturated = round(statistics.median(saturation_tflops), 3) if saturation_tflops else None

    # F_eff_weighted: FLOPs-weighted mean throughput in saturation regime.
    # Each shape's TFLOPS is weighted by its share of C_dense (N*K).
    # This is the most principled F_eff: it reflects the average throughput
    # the GPU delivers per FLOP of actual model compute.
    F_eff_weighted = None
    if sat_per_batch:
        weighted_vals: list[float] = []
        for batch_tflops in sat_per_batch.values():
            w_sum = sum(
                batch_tflops[name] * flops_weight[name]
                for name in batch_tflops
            )
            weighted_vals.append(w_sum / total_weight)
        F_eff_weighted = round(statistics.median(weighted_vals), 3)

    return {
        "F_eff_tflops_median": round(statistics.median(all_tflops), 3),
        "F_eff_tflops_transition": round(statistics.median(transition_tflops), 3),
        "F_eff_tflops_saturated": F_eff_saturated,
        "F_eff_tflops_weighted": F_eff_weighted,
        "flops_weights": {name: w for name, w in flops_weight.items()},
        "per_shape_B": per_shape_B,
    }


def parse_models(spec: str) -> list[tuple[str, int]]:
    result = []
    for entry in spec.split(","):
        entry = entry.strip()
        if ":" in entry:
            name, tp_str = entry.rsplit(":", 1)
            result.append((name.strip(), int(tp_str.strip())))
        else:
            result.append((entry, 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure GEMM effective TFLOPS")
    parser.add_argument(
        "--models",
        default="Qwen2.5-7B:1,Llama-3.1-8B-Instruct:1,Qwen2.5-14B:2",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: sd_toggle/configs/F_eff_bench_<gpu_short>.json "
             "(gpu_short auto-derived from device name).",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    host = socket.gethostname()
    # Derive short GPU tag: "NVIDIA A100-SXM4-80GB" → "a100",
    # "NVIDIA B200" → "b200".
    _tokens = gpu_name.replace("NVIDIA", "").strip().split()
    gpu_short = _tokens[0].split("-")[0].lower() if _tokens else "gpu"

    if args.output is None:
        args.output = f"sd_toggle/configs/F_eff_bench_{gpu_short}.json"

    print(f"GPU: {gpu_name}  (short tag: {gpu_short})")
    print(f"Host: {host}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Output: {args.output}")
    print()

    models_spec = parse_models(args.models)

    results: dict[str, Any] = {
        "gpu": gpu_name,
        "host": host,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "warmup_runs": WARMUP,
        "timed_runs": TIMED,
        "batch_sizes": BATCH_SIZES,
        "transition_batch_sizes": sorted(TRANSITION_BATCH_SIZES),
        "models": {},
    }

    all_saturated: list[float] = []
    all_weighted: list[float] = []

    for model_name, tp in models_spec:
        key = f"{model_name}_tp{tp}"
        print(f"=== {key} ===")
        data = measure_model(model_name, tp, device)
        results["models"][key] = data
        sat = data["F_eff_tflops_saturated"]
        wgt = data["F_eff_tflops_weighted"]
        if sat is not None:
            all_saturated.append(sat)
        if wgt is not None:
            all_weighted.append(wgt)
        print(
            f"  -> median={data['F_eff_tflops_median']:.1f} TFLOPS  "
            f"saturated={sat}  "
            f"weighted={wgt}\n"
        )

    # Cross-model hardware F_eff: median of per-model FLOPs-weighted saturation.
    # FLOPs-weighted mean accounts for the fact that FFN GEMMs dominate C_dense
    # (~75% of FLOPs) and achieve higher throughput than attention projections.
    # This gives the most principled single-number summary of realized throughput.
    if all_weighted:
        hw_F_eff = round(statistics.median(all_weighted), 3)
        results["F_eff_hardware_tflops"] = hw_F_eff
        print(f"=== Hardware F_eff (FLOPs-weighted, cross-model median) = {hw_F_eff:.1f} TFLOPS ===")
    elif all_saturated:
        hw_F_eff = round(statistics.median(all_saturated), 3)
        results["F_eff_hardware_tflops"] = hw_F_eff
        print(f"=== Hardware F_eff (unweighted fallback) = {hw_F_eff:.1f} TFLOPS ===")
    else:
        results["F_eff_hardware_tflops"] = None
        print("WARNING: No saturation data — add batch sizes >= 256")

    # Also store unweighted for comparison
    results["F_eff_unweighted_tflops"] = (
        round(statistics.median(all_saturated), 3) if all_saturated else None
    )

    results["saturation_batch_sizes"] = sorted(SATURATION_BATCH_SIZES)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
