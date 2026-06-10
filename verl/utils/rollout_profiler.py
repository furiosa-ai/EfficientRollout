"""Lightweight profiling for experiments.

Outputs:
  step_summary.jsonl    — one line per training step (phase breakdown)
  gpu_rollout_summary.jsonl — K lines per step (one per GPU/worker)

Enabled by default; overhead is ~1ms per step (one JSON serialize + file append).
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class StepSummary:
    """One line per training step."""

    step: int
    epoch: int
    step_time_sec: float
    rollout_time_sec: float
    reward_time_sec: float
    old_log_prob_time_sec: float
    ref_time_sec: float
    adv_time_sec: float
    update_actor_time_sec: float
    requantize_time_sec: float
    rollout_fraction: float  # rollout_time / step_time


@dataclass
class GpuRolloutSummary:
    """One line per GPU per step."""

    step: int = 0
    epoch: int = 0
    gpu_id: int = 0
    makespan_sec: float = 0.0
    n_requests: int = 0
    n_completed: int = 0
    mean_response_len: float = 0.0
    max_response_len: int = 0
    # SD fields
    sd_activated_at_tick: int = -1  # 0=always-on, >0=toggle tick, -1=disabled
    sd_activated_at_batch: int = -1  # active batch size when toggle fired, -1=disabled
    acceptance_rate: float = 0.0  # accepted/drafted tokens ratio (0-1)
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    num_drafts: int = 0
    accepted_per_pos: list[int] = field(default_factory=list)
    toggle_L_accept_used: float = 0.0  # L_accept fed into roofline toggle this rollout
    sd_current_gamma: int = -1         # current γ value (adaptive-γ) or -1 if n/a
    sd_elevation_to_gamma: int = -1    # transition target γ (γ_ladder[idx+1], -1 if last rung)
    sd_ar_observed: float = -1.0       # observed acceptance rate this rollout (AR-threshold refactor); -1.0 = no SD rollout yet
    sd_elevated_at_step: int = -1      # tick at last elevation (-1 if never)


class RolloutProfiler:
    """Lightweight JSONL profiler for RL training steps and per-GPU rollout summaries."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return
        os.makedirs(log_dir, exist_ok=True)
        self._step_path = os.path.join(log_dir, "step_summary.jsonl")
        self._gpu_path = os.path.join(log_dir, "gpu_rollout_summary.jsonl")

    def log_step_summary(self, step: int, epoch: int, timing_raw: dict) -> None:
        """Extract phase timings from trainer's timing_raw and dump one JSONL line.

        Args:
            step: global training step
            epoch: current epoch
            timing_raw: dict populated by marked_timer in ray_trainer.py
                        Keys: "step", "gen", "reward", "old_log_prob", "adv",
                              "update_actor", and possibly "RefPolicy", "values", etc.
        """
        if not self.enabled:
            return
        step_time = timing_raw.get("step", 0.0)
        rollout_time = timing_raw.get("gen", 0.0)
        entry = StepSummary(
            step=step,
            epoch=epoch,
            step_time_sec=round(step_time, 3),
            rollout_time_sec=round(rollout_time, 3),
            reward_time_sec=round(timing_raw.get("reward", 0.0), 3),
            old_log_prob_time_sec=round(timing_raw.get("old_log_prob", 0.0), 3),
            ref_time_sec=round(timing_raw.get("RefPolicy", 0.0), 3),
            adv_time_sec=round(timing_raw.get("adv", 0.0), 3),
            update_actor_time_sec=round(timing_raw.get("update_actor", 0.0), 3),
            requantize_time_sec=round(timing_raw.get("requantize", 0.0), 3),
            rollout_fraction=round(rollout_time / step_time, 3) if step_time > 0 else 0.0,
        )
        with open(self._step_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def log_gpu_rollout_summaries(
        self, step: int, epoch: int, summaries: list[GpuRolloutSummary]
    ) -> None:
        """Dump per-GPU rollout summaries for one training step.

        Args:
            step: global training step
            epoch: current epoch
            summaries: list of GpuRolloutSummary (one per GPU/worker)
        """
        if not self.enabled:
            return
        with open(self._gpu_path, "a") as f:
            for s in summaries:
                s.step = step
                s.epoch = epoch
                f.write(json.dumps(asdict(s)) + "\n")
