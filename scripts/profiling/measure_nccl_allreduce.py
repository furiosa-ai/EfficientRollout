"""Measure all-reduce latency for transformer-decode-sized tensors on 2-GPU NVLink.

Runs as a 2-process distributed job (one process per GPU).

Usage:
    torchrun --nproc_per_node=2 scripts/profiling/measure_nccl_allreduce.py \
        --output sd_toggle/configs/c_comm_bench.json

Fallback (if torchrun unavailable):
    python scripts/profiling/measure_nccl_allreduce.py --spawn
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WARMUP = 10
TIMED = 100
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]

MODEL_ARCH = {
    "Qwen2.5-7B": {"D": 3584, "L": 28},
    "Llama-3.1-8B-Instruct": {"D": 4096, "L": 32},
    "Qwen2.5-14B": {"D": 5120, "L": 48},
}


def measure_allreduce_latencies(rank: int, world_size: int) -> dict[str, float]:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    results: dict[str, float] = {}
    d_values = sorted(set(arch["D"] for arch in MODEL_ARCH.values()))

    for D in d_values:
        for B in BATCH_SIZES:
            tensor = torch.randn(B, D, dtype=torch.float16, device=device)

            for _ in range(WARMUP):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)

            times: list[float] = []
            for _ in range(TIMED):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                tensor = torch.randn(B, D, dtype=torch.float16, device=device)
                dist.barrier()
                start.record(torch.cuda.current_stream(device))
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                end.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
                times.append(start.elapsed_time(end) * 1e3)  # ms -> us

            median_us = statistics.median(times)
            key = f"D{D}_B{B}"
            results[key] = round(median_us, 3)

            if rank == 0:
                print(f"  D={D:5d} B={B:3d}: {median_us:7.2f} us  "
                      f"({B*D*2/1e6:.2f} MB payload)")

    return results


def compute_c_comm(latency_results: dict[str, float]) -> dict[str, Any]:
    c_comm: dict[str, Any] = {}

    for model_name, arch in MODEL_ARCH.items():
        D = arch["D"]
        L = arch["L"]
        key = f"{model_name}_tp2"
        c_comm[key] = {}

        for B in BATCH_SIZES:
            lat_key = f"D{D}_B{B}"
            if lat_key not in latency_results:
                continue
            lat_us = latency_results[lat_key]
            lat_sec = lat_us * 1e-6
            c_comm_sec = 2 * L * lat_sec
            c_comm[key][f"c_comm_B{B}_sec"] = round(c_comm_sec, 6)

        lat_b32_key = f"D{D}_B32"
        if lat_b32_key in latency_results:
            lat_b32_sec = latency_results[lat_b32_key] * 1e-6
            c_comm[key]["c_comm_B32_sec"] = round(2 * L * lat_b32_sec, 6)

        lat_b1_key = f"D{D}_B1"
        if lat_b1_key in latency_results:
            lat_b1_sec = latency_results[lat_b1_key] * 1e-6
            c_comm[key]["c_comm_B1_sec"] = round(2 * L * lat_b1_sec, 6)

    return c_comm


def run_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=world_size, rank=rank,
    )
    _run_measurements(rank, world_size, args)
    dist.destroy_process_group()


def _run_measurements(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        host = socket.gethostname()
        print(f"GPU: {gpu_name}  (world_size={world_size})")
        print(f"Host: {host}")
        print(f"PyTorch: {torch.__version__}")
        print()

    latency_results = measure_allreduce_latencies(rank, world_size)

    if rank == 0:
        c_comm = compute_c_comm(latency_results)

        gpu_name = torch.cuda.get_device_name(0)
        host = socket.gethostname()

        output: dict[str, Any] = {
            "gpu": f"{gpu_name} NVLink",
            "host": host,
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "world_size": world_size,
            "warmup_runs": WARMUP,
            "timed_runs": TIMED,
            "batch_sizes": BATCH_SIZES,
            "per_allreduce_latency_us": latency_results,
            "c_comm_per_model_tp2": c_comm,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {output_path}")

        print("\n=== c_comm summary ===")
        for model_key, vals in c_comm.items():
            b1 = vals.get("c_comm_B1_sec", float("nan"))
            b32 = vals.get("c_comm_B32_sec", float("nan"))
            print(f"  {model_key}: B1={b1*1e3:.3f} ms  B32={b32*1e3:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure NCCL all-reduce latency")
    parser.add_argument("--output", default="sd_toggle/configs/c_comm_bench.json")
    parser.add_argument("--spawn", action="store_true",
                        help="Use mp.spawn instead of torchrun (fallback mode)")
    args = parser.parse_args()

    if args.spawn:
        world_size = 2
        mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size == 1 and not os.environ.get("RANK"):
            print("Not launched via torchrun; falling back to mp.spawn with 2 processes.")
            mp.spawn(run_worker, args=(2, args), nprocs=2, join=True)
            return

        dist.init_process_group(backend="nccl", init_method="env://")
        _run_measurements(rank, world_size, args)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
