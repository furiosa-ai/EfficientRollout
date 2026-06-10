"""vLLM subprocess measurement module for component sweep.

Measures T_T (AR target decode), T_D (drafter decode), T_V (target verify)
via two-pass subprocess measurement with VLLM_SD_TIMING=1.

Public API:
    run_sweep(model, gpu, gammas, batches, seq_lens, ...) -> Path
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Canonical sweep grid — 9 batches × 5 seq_lens = 45 points per (mode, γ).
# Tuned to fit 7B/8B/14B at γ=15 within GPU_MEM_UTIL=0.80 without feasibility
# pre-filtering. (Previous ranges up to B=256, S=8192 were over-aggressive: they
# caused sweep OOMs on 7B/8B and triggered 21% pre-filtering on 14B.)
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64]
DEFAULT_BATCHES_DENSE = DEFAULT_BATCHES  # kept for backward compat
DEFAULT_SEQ_LENS = [256, 512, 1024, 2048, 4096]
DEFAULT_SEQ_LENS_DENSE = DEFAULT_SEQ_LENS  # kept for backward compat
DEFAULT_GAMMAS = [3, 7, 11, 15]
DEFAULT_GAMMAS_ALL = [3, 7, 11, 15]

GEN_TOKENS = 50
GPU_MEM_UTIL = 0.80
N_WARMUP = 5
N_REPEAT = 15

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "lv1_roofline"

# Per-model memory metadata — used for staggered worker launch (model load threshold).
# Feasibility pre-filtering was removed: sweep range is pre-tuned (B≤64, S≤4096)
# to fit all target models within GPU budget, eliminating the need for runtime
# feasibility checks. If a point OOMs, the subprocess fails and that point is
# simply absent from the CSV (same effect, without silent pre-filtering that
# hid systematic fit gaps — notably Qwen-14B losing 21% of its sweep domain).
MODEL_KV: dict[str, dict[str, float]] = {
    "Qwen/Qwen2.5-7B": {"kv_per_tok": 57344, "model_gb": 14.2, "drafter_gb": 5.1},
    "Qwen2.5-7B": {"kv_per_tok": 57344, "model_gb": 14.2, "drafter_gb": 5.1},
    "Qwen/Qwen2.5-14B": {"kv_per_tok": 196608, "model_gb": 27.5, "drafter_gb": 9.0},
    "Qwen2.5-14B": {"kv_per_tok": 196608, "model_gb": 27.5, "drafter_gb": 9.0},
    "meta-llama/Meta-Llama-3.1-8B": {"kv_per_tok": 131072, "model_gb": 15.0, "drafter_gb": 5.2},
    "LLaMA3.1-8B": {"kv_per_tok": 131072, "model_gb": 15.0, "drafter_gb": 5.2},
    "meta-llama/Llama-3.1-8B-Instruct": {"kv_per_tok": 131072, "model_gb": 15.0, "drafter_gb": 5.2},
    "Llama-3.1-8B-Instruct": {"kv_per_tok": 131072, "model_gb": 15.0, "drafter_gb": 5.2},
}


def _build_subprocess_script(
    model: str,
    gpu: int,
    gamma: int,
    batches: list[int],
    seq_lens: list[int],
    tp: int = 1,
    gen_tokens: int = GEN_TOKENS,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
) -> str:
    """Build the Python script to run inside a subprocess."""
    max_seq = max(seq_lens) + gen_tokens + 256
    has_drafter = gamma > 0

    # Full Cartesian product of (batch, seq). Feasibility pre-filtering was
    # removed — calibrate scripts use pre-tuned ranges (max B=64, max S=4096)
    # that fit all target models without dropping any point. If a corner OOMs,
    # it simply fails subprocess and is absent from CSV.
    configs = [(b, s) for b in batches for s in seq_lens]
    configs_str = repr(configs)

    spec_config_str = "None"
    if gamma > 0:
        spec_config_str = f'{{"method": "quant_self", "num_speculative_tokens": {gamma}}}'

    gpu_ids = ",".join(str(gpu + i) for i in range(tp))
    timing_file = f"/tmp/sd_timing_output_gpu{gpu}.txt"

    script = f'''
import os, time, gc, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_ids}"
os.environ["VLLM_SD_TIMING"] = "1"
os.environ["VLLM_SD_TIMING_FILE"] = "{timing_file}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_gpu{gpu}"

from vllm import LLM, SamplingParams

spec_config = {spec_config_str}

llm = LLM(
    model="{model}",
    tensor_parallel_size={tp},
    gpu_memory_utilization={GPU_MEM_UTIL},
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len={max_seq},
    max_num_batched_tokens=16384,
    speculative_config=spec_config,
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("{model}", trust_remote_code=True)

base_text = "The quick brown fox jumps over the lazy dog. " * 200
base_ids = tokenizer.encode(base_text)

configs = {configs_str}

for batch, seq in configs:
    prompt_ids = base_ids[:seq]
    if len(prompt_ids) < seq:
        prompt_ids = (prompt_ids * (seq // len(prompt_ids) + 1))[:seq]
    prompt_text = tokenizer.decode(prompt_ids)
    prompts = [prompt_text] * batch

    sp = SamplingParams(
        temperature=0.0,
        max_tokens={gen_tokens},
        min_tokens={gen_tokens},
        ignore_eos=True,
    )

    # OOM-tolerant: any failure at this (batch, seq) is logged and skipped so
    # the remaining grid continues to run. Replaces the old _is_feasible
    # pre-filter which hid systematic coverage gaps (notably 14B, 21% silent drop).
    try:
        for _ in range({n_warmup}):
            llm.generate(prompts, sp)

        for rep in range({n_repeat}):
            _meta_line = f"SWEEP_META,batch={{batch}},seq={{seq}},rep={{rep}},gamma={gamma}"
            print(_meta_line, flush=True)
            _tf = os.environ.get("VLLM_SD_TIMING_FILE", "")
            if _tf:
                with open(_tf, "a") as _ff:
                    _ff.write(_meta_line + "\\n")
            llm.generate(prompts, sp)
    except Exception as _sd_exc:
        _reason = f"{{type(_sd_exc).__name__}}:{{str(_sd_exc)[:160]}}"
        _oom_line = f"SWEEP_SKIP,batch={{batch}},seq={{seq}},gamma={gamma},reason={{_reason}}"
        print(_oom_line, flush=True)
        sys.stderr.write(_oom_line + "\\n")
        sys.stderr.flush()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("SWEEP_DONE", flush=True)
'''
    return script


def _parse_timing_output(stdout: str, gamma: int) -> list[dict]:
    """Parse SD_TIMING lines from combined subprocess output into structured records."""
    records = []
    current_meta: dict[str, int] = {"batch": 0, "seq": 0, "rep": 0, "gamma": gamma}

    for line in stdout.split("\n"):
        # Strip ANSI codes and vLLM worker prefix
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        line = re.sub(r"\(EngineCore_DP0 pid=\d+\)\s*", "", line)
        line = line.strip()

        if line.startswith("SWEEP_META,"):
            parts = dict(p.split("=") for p in line.split(",")[1:])
            current_meta = {
                "batch": int(parts["batch"]),
                "seq": int(parts["seq"]),
                "rep": int(parts["rep"]),
                "gamma": int(parts["gamma"]),
            }
            continue

        if not line.startswith("SD_TIMING,"):
            continue

        parts = line.split(",")
        phase = parts[1]
        kvs: dict[str, float | str] = {}
        for p in parts[2:]:
            if "=" in p:
                k, v = p.split("=", 1)
                try:
                    kvs[k] = float(v)
                except ValueError:
                    kvs[k] = v

        record: dict = {**current_meta}

        if phase == "T_T":
            record["mode"] = "target_decode"
            record["elapsed_ms"] = kvs.get("elapsed_ms", 0)
            record["batch_reported"] = int(kvs.get("batch", current_meta["batch"]))  # type: ignore[arg-type]
            record["num_tokens"] = int(kvs.get("num_tokens", 0))  # type: ignore[arg-type]
        elif phase == "T_T_sd":
            record["mode"] = "target_decode_sd"
            record["elapsed_ms"] = kvs.get("elapsed_ms", 0)
            record["batch_reported"] = int(kvs.get("batch", current_meta["batch"]))  # type: ignore[arg-type]
        elif phase == "T_V":
            record["mode"] = "target_verify"
            record["elapsed_ms"] = kvs.get("elapsed_ms", 0)
            record["batch_reported"] = int(kvs.get("batch", current_meta["batch"]))  # type: ignore[arg-type]
            record["num_tokens"] = int(kvs.get("num_tokens", 0))  # type: ignore[arg-type]
            record["qlen"] = int(kvs.get("qlen", gamma + 1))  # type: ignore[arg-type]
        elif phase == "T_D":
            record["mode"] = "drafter_decode"
            record["fwd_ms"] = kvs.get("fwd_ms", 0)
            record["full_ms"] = kvs.get("full_ms", 0)
            record["overhead_ms"] = kvs.get("overhead_ms", 0)
            record["elapsed_ms"] = kvs.get("fwd_ms", 0)  # use fwd_ms as primary
            record["batch_reported"] = int(kvs.get("batch", current_meta["batch"]))  # type: ignore[arg-type]
        else:
            continue

        records.append(record)

    return records


def _aggregate_records(records: list[dict]) -> list[dict]:
    """Aggregate repeated measurements: discard first 20%, compute mean/median/std/min/max."""
    import statistics

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r["batch"], r["seq"], r["gamma"], r["mode"])
        groups[key].append(r["elapsed_ms"])

    aggregated = []
    for (batch, seq, gamma, mode), values in sorted(groups.items()):
        if len(values) < 3:
            continue
        n_discard = max(1, len(values) // 5)
        values = values[n_discard:]
        if len(values) < 2:
            continue

        aggregated.append({
            "batch": batch,
            "seq": seq,
            "gamma": gamma,
            "mode": mode,
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_ms": min(values),
            "max_ms": max(values),
            "n_samples": len(values),
        })

    return aggregated


def _run_pass(
    model: str,
    gpu: int,
    gamma: int,
    batches: list[int],
    seq_lens: list[int],
    tp: int = 1,
    gen_tokens: int = GEN_TOKENS,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
) -> list[dict]:
    """Run one measurement pass (AR gamma=0 or SD gamma>0) and return aggregated records."""
    mode_name = "AR" if gamma == 0 else f"SD-gamma{gamma}"
    print(f"\n{'='*60}")
    print(f"Pass: {mode_name} | Model: {model} | GPU: {gpu} | TP: {tp}")
    print(f"Grid: {len(batches)} batches x {len(seq_lens)} seqs")
    print(f"Warmup: {n_warmup}, Repeat: {n_repeat}")
    print(f"{'='*60}\n")

    script = _build_subprocess_script(
        model, gpu, gamma, batches, seq_lens, tp, gen_tokens, n_warmup, n_repeat
    )

    env = os.environ.copy()
    gpu_ids = ",".join(str(gpu + i) for i in range(tp))
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["VLLM_SD_TIMING"] = "1"
    # Use GPU-specific timing file to avoid conflicts during parallel sweeps
    timing_file = f"/tmp/sd_timing_output_gpu{gpu}.txt"
    env["VLLM_SD_TIMING_FILE"] = timing_file
    if os.path.exists(timing_file):
        os.remove(timing_file)

    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=7200,
    )
    elapsed = time.time() - start

    if proc.returncode != 0:
        print(f"ERROR: Subprocess failed (exit {proc.returncode})")
        print(f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}")
        return []

    print(f"Pass completed in {elapsed:.0f}s")

    combined_output = ""
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            combined_output = f.read()
        os.remove(timing_file)
        print(f"Read {len(combined_output)} chars from timing file")

    combined_output += "\n" + proc.stdout + "\n" + proc.stderr

    records = _parse_timing_output(combined_output, gamma)
    print(f"Parsed {len(records)} raw timing records")

    aggregated = _aggregate_records(records)
    print(f"Aggregated to {len(aggregated)} measurement points")

    return aggregated


def run_sweep(
    model: str,
    gpu: int,
    gammas: list[int] | None = None,
    batches: list[int] | None = None,
    seq_lens: list[int] | None = None,
    tp: int = 1,
    gen_tokens: int = GEN_TOKENS,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
    output: Path | str | None = None,
) -> Path:
    """Run full component sweep and save results to CSV.

    Two-pass measurement:
      Pass 1: AR engine (gamma=0) → T_T
      Pass 2+: SD engine (gamma=N) → T_D, T_V

    Args:
        model: HuggingFace model ID.
        gpu: GPU index for CUDA_VISIBLE_DEVICES.
        gammas: SD gamma values to sweep. Defaults to [5].
        batches: Batch sizes. Defaults to DEFAULT_BATCHES.
        seq_lens: Sequence lengths. Defaults to DEFAULT_SEQ_LENS.
        tp: Tensor parallelism degree.
        gen_tokens: Number of tokens to generate per request.
        n_warmup: Warmup iterations (discarded).
        n_repeat: Measurement repetitions.
        output: Output CSV path. Auto-generated from model name if None.

    Returns:
        Path to the saved CSV file.
    """
    if gammas is None:
        gammas = DEFAULT_GAMMAS
    if batches is None:
        batches = DEFAULT_BATCHES
    if seq_lens is None:
        seq_lens = DEFAULT_SEQ_LENS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output is None:
        model_short = model.split("/")[-1].replace(".", "").replace("-", "").lower()
        output_path = RESULTS_DIR / f"{model_short}_components.csv"
    else:
        output_path = Path(output)

    all_records: list[dict] = []

    # Pass 1: AR (gamma=0) → T_T
    print("\n" + "=" * 70)
    print("PHASE 1: AR Pass (T_T measurement)")
    print("=" * 70)
    ar_records = _run_pass(
        model, gpu, gamma=0,
        batches=batches, seq_lens=seq_lens, tp=tp,
        gen_tokens=gen_tokens, n_warmup=n_warmup, n_repeat=n_repeat,
    )
    all_records.extend(ar_records)

    # Pass 2+: SD passes → T_D, T_V
    for gamma in gammas:
        print("\n" + "=" * 70)
        print(f"PHASE 2: SD Pass gamma={gamma} (T_D, T_V measurement)")
        print("=" * 70)
        sd_records = _run_pass(
            model, gpu, gamma=gamma,
            batches=batches, seq_lens=seq_lens, tp=tp,
            gen_tokens=gen_tokens, n_warmup=n_warmup, n_repeat=n_repeat,
        )
        all_records.extend(sd_records)

    # Save CSV
    fieldnames = [
        "model", "batch", "seq", "gamma", "mode",
        "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "n_samples",
    ]
    if all_records:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            model_short_name = model.split("/")[-1]
            for r in all_records:
                row = {"model": model_short_name, **r}
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"\nSaved {len(all_records)} records to {output_path}")
    else:
        print("WARNING: No records to save!")

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    modes: dict[str, int] = defaultdict(int)
    for r in all_records:
        modes[r["mode"]] += 1
    for mode, count in sorted(modes.items()):
        print(f"  {mode}: {count} points")
    print(f"  Total: {len(all_records)} points")
    print(f"  Output: {output_path}")

    return output_path


def _wait_for_gpu_memory(gpu: int, threshold_mb: int = 14000,
                         timeout: int = 300, poll_interval: int = 5) -> bool:
    """Poll nvidia-smi until GPU has loaded model weights above threshold."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu}",
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            mem_mb = int(result.stdout.strip())
            if mem_mb >= threshold_mb:
                return True
        except (ValueError, subprocess.TimeoutExpired):
            pass
        time.sleep(poll_interval)
    return False


def run_sweep_multi_gpu(
    model: str,
    gpus: list[int],
    gammas: list[int] | None = None,
    batches: list[int] | None = None,
    seq_lens: list[int] | None = None,
    tp: int = 1,
    gen_tokens: int = GEN_TOKENS,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
    output: Path | str | None = None,
) -> Path:
    """Run sweep across multiple GPUs in parallel.

    Key improvements over single-GPU run_sweep():
    - AR pass (T_T) runs ONCE, not repeated per gamma
    - All passes (AR + SD per gamma) run in parallel
    - Staggered launch: models load one at a time (avoids disk I/O
      contention), but measurements overlap across GPUs
    - Grid split across GPUs when more GPUs than passes

    With 10 GPUs and gammas=[3,7,11,15]:
      5 passes (AR + 4×SD) × 2 GPU shards = 10 parallel workers.
      Each shard gets half the batch sizes.

    Args:
        model: HuggingFace model ID.
        gpus: List of physical GPU indices (e.g. [0,1,2,3,4,5,6,7]).
        gammas: SD gamma values. Defaults to [3, 7, 11, 15].
        batches: Batch sizes. Defaults to DEFAULT_BATCHES_DENSE.
        seq_lens: Sequence lengths. Defaults to DEFAULT_SEQ_LENS_DENSE.
        tp: Tensor parallelism degree.
        output: Output CSV path.

    Returns:
        Path to merged CSV.
    """
    import threading

    if gammas is None:
        gammas = DEFAULT_GAMMAS_ALL
    if batches is None:
        batches = DEFAULT_BATCHES_DENSE
    if seq_lens is None:
        seq_lens = DEFAULT_SEQ_LENS_DENSE

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output is None:
        model_short = model.split("/")[-1].replace(".", "").replace("-", "").lower()
        output_path = RESULTS_DIR / f"{model_short}_components_multi.csv"
    else:
        output_path = Path(output)

    # Build pass list: AR (gamma=0) + SD per gamma
    pass_gammas = [0] + list(gammas)
    n_passes = len(pass_gammas)
    n_gpus = len(gpus)

    # Assign GPUs to passes, split grid if extras available
    gpus_per_pass = n_gpus // n_passes
    extra_gpus = n_gpus % n_passes

    tasks: list[dict] = []
    gpu_idx = 0

    for i, gamma in enumerate(pass_gammas):
        n_gpu_this = gpus_per_pass + (1 if i < extra_gpus else 0)
        pass_gpu_list = gpus[gpu_idx:gpu_idx + n_gpu_this]
        gpu_idx += n_gpu_this

        if n_gpu_this <= 1:
            tasks.append({
                "gamma": gamma, "gpu": pass_gpu_list[0],
                "batches": batches, "seq_lens": seq_lens,
                "label": f"{'AR' if gamma == 0 else f'SD-γ{gamma}'} [full]",
            })
        else:
            chunk = (len(batches) + n_gpu_this - 1) // n_gpu_this
            for j, g in enumerate(pass_gpu_list):
                b_slice = batches[j * chunk:(j + 1) * chunk]
                if not b_slice:
                    continue
                tasks.append({
                    "gamma": gamma, "gpu": g, "batches": b_slice,
                    "seq_lens": seq_lens,
                    "label": f"{'AR' if gamma == 0 else f'SD-γ{gamma}'} B={b_slice[0]}..{b_slice[-1]}",
                })

    # Print plan
    print(f"\n{'=' * 70}")
    print("MULTI-GPU SWEEP PLAN (staggered launch)")
    print(f"{'=' * 70}")
    print(f"Model: {model} | TP: {tp}")
    print(f"GPUs: {gpus} ({n_gpus} total)")
    print(f"Passes: AR + SD-g{','.join(str(g) for g in gammas)}")
    print(f"Grid: {len(batches)} batches x {len(seq_lens)} seqs")
    print(f"Workers: {len(tasks)}")
    for t in tasks:
        print(f"  GPU {t['gpu']:>2}: {t['label']} ({len(t['batches'])}x{len(t['seq_lens'])})")
    print(f"{'=' * 70}\n")

    # Staggered launch: start each worker thread, wait for its GPU to load
    # before starting the next. This serializes model loading (I/O bound)
    # while allowing measurements to overlap (GPU bound).
    all_records: list[dict] = []
    errors: list[tuple[dict, Exception]] = []
    lock = threading.Lock()

    def run_task_thread(task: dict) -> None:
        try:
            records = _run_pass(
                model, task["gpu"], task["gamma"],
                task["batches"], task["seq_lens"], tp,
                gen_tokens, n_warmup, n_repeat,
            )
            with lock:
                all_records.extend(records)
            print(f"  done GPU {task['gpu']:>2}: {task['label']} -- {len(records)} points")
        except Exception as e:
            with lock:
                errors.append((task, e))
            print(f"  FAIL GPU {task['gpu']:>2}: {task['label']} -- {e}")

    threads: list[threading.Thread] = []

    # Derive per-GPU memory threshold for staggered launch. Model weight is
    # sharded across `tp` GPUs, so each primary GPU only holds model_gb/tp.
    # (Previously threshold was computed per-pair, which timed out for tp>1
    # and degraded the stagger to 10s-apart launches → NCCL init race.)
    spec = MODEL_KV.get(model) or MODEL_KV.get(model.split("/")[-1])
    load_threshold_mb = int(spec["model_gb"] / tp * 0.8 * 1024) if spec else 14000

    # NCCL/compile init races for tp>1 need a longer gap between workers.
    stagger_s = 30 if tp > 1 else 10

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] Launching GPU {task['gpu']}: {task['label']}")
        t = threading.Thread(target=run_task_thread, args=(task,), daemon=True)
        t.start()
        threads.append(t)

        # Stagger strategy:
        # - First worker: wait for full model load (populates OS page cache
        #   + torch.compile cache). Subsequent workers reuse both.
        # - Remaining workers: stagger_s delay between launches to let NCCL
        #   init + torch.compile settle before the next worker starts.
        if i == 0:
            loaded = _wait_for_gpu_memory(task["gpu"], threshold_mb=load_threshold_mb)
            if loaded:
                print(f"         GPU {task['gpu']} loaded (page cache + compile cache warm). Launching rest.")
            else:
                print(f"         GPU {task['gpu']} load timeout. Launching rest anyway.")
        elif i < len(tasks) - 1:
            time.sleep(stagger_s)

    print(f"\n  All {len(tasks)} workers launched. Waiting for measurements...\n")

    for t in threads:
        t.join()

    if errors:
        print(f"\nWARNING: {len(errors)}/{len(tasks)} tasks failed!")

    # Save merged CSV
    fieldnames = [
        "model", "batch", "seq", "gamma", "mode",
        "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "n_samples",
    ]
    if all_records:
        model_short_name = model.split("/")[-1]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(all_records, key=lambda x: (x["gamma"], x["mode"], x["batch"], x["seq"])):
                row = {"model": model_short_name, **r}
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"\nSaved {len(all_records)} records to {output_path}")
    else:
        print("WARNING: No records collected!")

    # Summary
    print(f"\n{'=' * 70}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 70}")
    modes: dict[str, int] = defaultdict(int)
    for r in all_records:
        modes[r["mode"]] += 1
    for mode, count in sorted(modes.items()):
        print(f"  {mode}: {count} points")
    print(f"  Total: {len(all_records)} points")
    print(f"  Output: {output_path}")

    return output_path
