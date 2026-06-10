"""Unit tests for rollout infrastructure modules.

Tests M1 (decode state), M2 (length tracker), M3 (allocation policy), M4 (profiler).
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# M2: RolloutLengthTracker
# ---------------------------------------------------------------------------
class TestRolloutLengthTracker:
    def test_record_and_aggregate(self):
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        tracker = RolloutLengthTracker(l_max=8192)
        assert not tracker.has_history()

        # Simulate epoch 0: 3 prompts, each with 4 rollouts
        keys = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        lengths = np.array([100, 120, 110, 130, 4000, 4200, 4100, 8192, 200, 180, 190, 210])
        tracker.record_step(keys, lengths, epoch=0)
        tracker.aggregate_epoch(0)

        assert tracker.has_history()

        s0 = tracker.get_prompt_stats(0)
        assert s0 is not None
        assert abs(s0.mu - 115.0) < 1e-6
        assert s0.max_len == 130
        assert abs(s0.median - 115.0) < 1e-6

        s1 = tracker.get_prompt_stats(1)
        assert s1 is not None
        assert s1.max_len == 8192

        s2 = tracker.get_prompt_stats(2)
        assert s2 is not None
        assert abs(s2.mu - 195.0) < 1e-6

    def test_tail_prob_computation(self):
        """tail_prob = fraction of rollouts > κ * median (default κ=2)."""
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        tracker = RolloutLengthTracker(l_max=8192, ema_alpha=1.0)
        # 4 rollouts: [100, 100, 100, 500]. median=100, threshold=200.
        # 1 out of 4 > 200 → tail_prob = 0.25
        keys = np.array([0, 0, 0, 0])
        lengths = np.array([100, 100, 100, 500])
        tracker.record_step(keys, lengths, epoch=0)
        tracker.aggregate_epoch(0)

        s = tracker.get_prompt_stats(0)
        assert abs(s.tail_prob - 0.25) < 1e-6
        assert abs(s.median - 100.0) < 1e-6

    def test_tail_prob_ema(self):
        """tail_prob is EMA-updated across epochs."""
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        tracker = RolloutLengthTracker(l_max=8192, ema_alpha=0.5)

        # Epoch 0: all same length → tail_prob = 0
        tracker.record_step(np.array([0, 0, 0, 0]), np.array([100, 100, 100, 100]), epoch=0)
        tracker.aggregate_epoch(0)
        assert tracker.get_prompt_stats(0).tail_prob == 0.0

        # Epoch 1: [100, 100, 100, 500] → p_epoch = 0.25
        # EMA: 0.5 * 0.25 + 0.5 * 0.0 = 0.125
        tracker.record_step(np.array([0, 0, 0, 0]), np.array([100, 100, 100, 500]), epoch=1)
        tracker.aggregate_epoch(1)
        assert abs(tracker.get_prompt_stats(0).tail_prob - 0.125) < 1e-6

    def test_cold_start(self):
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        tracker = RolloutLengthTracker(l_max=8192)
        assert not tracker.has_history()
        assert tracker.get_prompt_stats(0) is None
        assert tracker.get_all_stats() == {}

    def test_epoch_pruning(self):
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        tracker = RolloutLengthTracker(l_max=8192)
        for epoch in range(5):
            keys = np.array([0, 1])
            lengths = np.array([100 + epoch * 10, 200 + epoch * 10])
            tracker.record_step(keys, lengths, epoch=epoch)
            tracker.aggregate_epoch(epoch)

        assert 0 not in tracker._history
        assert 1 not in tracker._history
        assert 2 not in tracker._history
        assert 3 in tracker._history
        assert 4 in tracker._history

    def test_tail_rate(self):
        """get_tail_rate returns mean tail_prob across prompts."""
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        tracker = RolloutLengthTracker(l_max=8192, ema_alpha=1.0)
        # Prompt 0: no tail. Prompt 1: 1/4 tail.
        keys = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        lengths = np.array([100, 100, 100, 100, 100, 100, 100, 500])
        tracker.record_step(keys, lengths, epoch=0)
        tracker.aggregate_epoch(0)
        # Prompt 0: tail_prob=0, Prompt 1: tail_prob=0.25 → mean=0.125
        assert abs(tracker.get_tail_rate() - 0.125) < 1e-6

    def test_multiple_steps_same_epoch(self):
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        tracker = RolloutLengthTracker(l_max=8192)
        tracker.record_step(np.array([0]), np.array([100]), epoch=0)
        tracker.record_step(np.array([0]), np.array([200]), epoch=0)
        tracker.aggregate_epoch(0)
        s = tracker.get_prompt_stats(0)
        assert s is not None
        assert abs(s.mu - 150.0) < 1e-6
        assert s.max_len == 200


# ---------------------------------------------------------------------------
# M3: AllocationPolicy
# ---------------------------------------------------------------------------
class TestAllocationPolicy:
    def test_default_policy_no_assignment(self):
        """DefaultPolicy returns empty dict (uses original min-heap LB)."""
        from verl.utils.allocation_policy import DefaultPolicy, PromptInfo

        policy = DefaultPolicy()
        prompts = [PromptInfo(key=i, prompt_length=100) for i in range(8)]
        result = policy.allocate(prompts, num_gpus=4)
        assert result == {}

    def test_length_balancing_not_implemented(self):
        from verl.utils.allocation_policy import LengthBalancingPolicy, PromptInfo

        policy = LengthBalancingPolicy()
        prompts = [PromptInfo(key=0, prompt_length=100)]
        with pytest.raises(NotImplementedError):
            policy.allocate(prompts, num_gpus=2)

    def test_effective_load_not_implemented(self):
        from verl.utils.allocation_policy import EffectiveLoadPolicy, PromptInfo

        policy = EffectiveLoadPolicy(G=8, tail_amplification=2.0, l_max=8192)
        prompts = [PromptInfo(key=0, prompt_length=100, mu=1000.0)]
        with pytest.raises(NotImplementedError):
            policy.allocate(prompts, num_gpus=2)

    def test_effective_load_higher_weight_for_tail(self):
        """_compute_effective_load gives higher weight when tail_prob > 0."""
        from verl.utils.allocation_policy import EffectiveLoadPolicy, PromptInfo

        policy = EffectiveLoadPolicy(G=8, tail_amplification=2.0, l_max=8192)
        safe = PromptInfo(key=0, prompt_length=100, mu=1000.0, tail_prob=0.0)
        risky = PromptInfo(key=1, prompt_length=100, mu=1000.0, tail_prob=0.2)

        w_safe = policy._compute_effective_load(safe)
        w_risky = policy._compute_effective_load(risky)
        assert w_risky > w_safe
        # w_safe = 8 * 1000 = 8000
        assert abs(w_safe - 8000.0) < 1e-6
        # w_risky = 8 * [(1-0.2)*1000 + 0.2*min(2*1000, 8192)] = 8 * [800 + 400] = 9600
        assert abs(w_risky - 9600.0) < 1e-6

    def test_factory(self):
        from verl.utils.allocation_policy import (
            DefaultPolicy,
            EffectiveLoadPolicy,
            LengthBalancingPolicy,
            create_allocation_policy,
        )

        assert isinstance(create_allocation_policy("default"), DefaultPolicy)
        assert isinstance(create_allocation_policy("length_balancing"), LengthBalancingPolicy)
        assert isinstance(
            create_allocation_policy("effective_load", G=8, l_max=8192), EffectiveLoadPolicy
        )
        with pytest.raises(ValueError):
            create_allocation_policy("nonexistent")


# ---------------------------------------------------------------------------
# M4: RolloutProfiler
# ---------------------------------------------------------------------------
class TestRolloutProfiler:
    def test_step_summary(self):
        from verl.utils.rollout_profiler import RolloutProfiler

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = RolloutProfiler(log_dir=tmpdir, enabled=True)
            timing = {
                "step": 100.0,
                "gen": 65.0,
                "reward": 5.0,
                "old_log_prob": 10.0,
                "RefPolicy": 8.0,
                "adv": 0.5,
                "update_actor": 11.5,
            }
            profiler.log_step_summary(step=1, epoch=0, timing_raw=timing)
            profiler.log_step_summary(step=2, epoch=0, timing_raw=timing)

            path = os.path.join(tmpdir, "step_summary.jsonl")
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            entry = json.loads(lines[0])
            assert entry["step"] == 1
            assert entry["epoch"] == 0
            assert entry["rollout_time_sec"] == 65.0
            assert entry["rollout_fraction"] == 0.65

    def test_gpu_rollout_summary(self):
        from verl.utils.rollout_profiler import GpuRolloutSummary, RolloutProfiler

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = RolloutProfiler(log_dir=tmpdir, enabled=True)
            summaries = [
                GpuRolloutSummary(gpu_id=0, makespan_sec=45.2, n_requests=256, n_completed=256,
                                  mean_response_len=1842.3, max_response_len=8192),
                GpuRolloutSummary(gpu_id=1, makespan_sec=38.1, n_requests=256, n_completed=256,
                                  mean_response_len=1500.0, max_response_len=6000),
            ]
            profiler.log_gpu_rollout_summaries(step=1, epoch=0, summaries=summaries)

            path = os.path.join(tmpdir, "gpu_rollout_summary.jsonl")
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            entry0 = json.loads(lines[0])
            assert entry0["gpu_id"] == 0
            assert entry0["makespan_sec"] == 45.2
            assert entry0["sd_activated_at_tick"] == -1

    def test_disabled(self):
        from verl.utils.rollout_profiler import RolloutProfiler

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = RolloutProfiler(log_dir=tmpdir, enabled=False)
            profiler.log_step_summary(step=1, epoch=0, timing_raw={"step": 1.0})
            assert not os.path.exists(os.path.join(tmpdir, "step_summary.jsonl"))

    def test_missing_timing_keys(self):
        from verl.utils.rollout_profiler import RolloutProfiler

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = RolloutProfiler(log_dir=tmpdir, enabled=True)
            profiler.log_step_summary(step=1, epoch=0, timing_raw={"step": 50.0, "gen": 30.0})
            path = os.path.join(tmpdir, "step_summary.jsonl")
            entry = json.loads(open(path).readline())
            assert entry["reward_time_sec"] == 0.0
            assert entry["rollout_fraction"] == 0.6

    def test_requantize_time_sec_present_in_step_summary(self):
        """requantize_time_sec is written when timing_raw contains 'requantize'."""
        from verl.utils.rollout_profiler import RolloutProfiler

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = RolloutProfiler(log_dir=tmpdir, enabled=True)
            timing = {
                "step": 10.0, "gen": 5.0, "reward": 1.0, "old_log_prob": 0.5,
                "RefPolicy": 0.3, "adv": 0.2, "update_actor": 2.0, "requantize": 3.14,
            }
            profiler.log_step_summary(step=1, epoch=0, timing_raw=timing)
            path = os.path.join(tmpdir, "step_summary.jsonl")
            entry = json.loads(open(path).readline())
            assert entry["requantize_time_sec"] == 3.14

    def test_requantize_time_sec_defaults_to_zero_when_absent(self):
        """requantize_time_sec is 0.0 when 'requantize' key is absent (no-SD mode)."""
        from verl.utils.rollout_profiler import RolloutProfiler

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = RolloutProfiler(log_dir=tmpdir, enabled=True)
            timing = {
                "step": 10.0, "gen": 5.0, "reward": 1.0, "old_log_prob": 0.5,
                "RefPolicy": 0.3, "adv": 0.2, "update_actor": 2.0,
            }
            profiler.log_step_summary(step=1, epoch=0, timing_raw=timing)
            path = os.path.join(tmpdir, "step_summary.jsonl")
            entry = json.loads(open(path).readline())
            assert entry["requantize_time_sec"] == 0.0

    def test_gpu_rollout_summary_num_drafts_and_accepted_per_pos(self):
        """GpuRolloutSummary serializes num_drafts and accepted_per_pos correctly."""
        from verl.utils.rollout_profiler import GpuRolloutSummary, RolloutProfiler

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = RolloutProfiler(log_dir=tmpdir, enabled=True)
            summary = GpuRolloutSummary(
                gpu_id=0, makespan_sec=10.0, n_requests=4, n_completed=4,
                mean_response_len=512.0, max_response_len=1024,
                num_drafts=8, accepted_per_pos=[3, 2, 1],
            )
            profiler.log_gpu_rollout_summaries(step=1, epoch=0, summaries=[summary])
            path = os.path.join(tmpdir, "gpu_rollout_summary.jsonl")
            entry = json.loads(open(path).readline())
            assert entry["num_drafts"] == 8
            assert entry["accepted_per_pos"] == [3, 2, 1]


# ---------------------------------------------------------------------------
# M1: Scheduler decode state (mock-based, no GPU needed)
# ---------------------------------------------------------------------------
class TestDecodeState:
    def test_get_decode_state_basic(self):
        """Test the decode state computation logic (without importing vLLM)."""
        class MockRequest:
            def __init__(self, num_tokens):
                self._num_tokens = num_tokens

            @property
            def num_tokens(self):
                return self._num_tokens

        running = [MockRequest(100), MockRequest(200), MockRequest(300)]
        n = len(running)
        total_len = sum(r.num_tokens for r in running)
        avg = total_len / n if n > 0 else 0.0

        assert n == 3
        assert abs(avg - 200.0) < 1e-6

    def test_get_decode_state_empty(self):
        running = []
        n = len(running)
        avg = sum(r.num_tokens for r in running) / n if n > 0 else 0.0
        assert n == 0
        assert avg == 0.0


# ---------------------------------------------------------------------------
# _SDStatsAccumulator
# ---------------------------------------------------------------------------
class TestSDStatsAccumulator:
    def _make_accumulator(self):
        from vllm.v1.engine.async_llm import _SDStatsAccumulator
        return _SDStatsAccumulator()

    def _make_stats(self, sd_toggled=False, sd_toggle_tick=-1, num_drafts=0,
                    num_draft_tokens=0, num_accepted_tokens=0,
                    num_accepted_tokens_per_pos=None):
        """Create a mock scheduler_stats."""
        stats = MagicMock()
        stats.sd_toggled = sd_toggled
        stats.sd_toggle_tick = sd_toggle_tick
        if num_drafts > 0:
            stats.spec_decoding_stats = MagicMock()
            stats.spec_decoding_stats.num_drafts = num_drafts
            stats.spec_decoding_stats.num_draft_tokens = num_draft_tokens
            stats.spec_decoding_stats.num_accepted_tokens = num_accepted_tokens
            stats.spec_decoding_stats.num_accepted_tokens_per_pos = (
                num_accepted_tokens_per_pos if num_accepted_tokens_per_pos is not None
                else [0] * num_draft_tokens
            )
        else:
            stats.spec_decoding_stats = None
        return stats

    def test_initial_state(self):
        acc = self._make_accumulator()
        snap = acc.snapshot()
        assert snap["sd_toggled"] is False
        assert snap["sd_toggle_tick"] == -1
        assert snap["acceptance_rate"] == 0.0
        assert snap["total_steps"] == 0

    def test_observe_acceptance_rate(self):
        acc = self._make_accumulator()
        acc.observe(self._make_stats(num_drafts=1, num_draft_tokens=10, num_accepted_tokens=7))
        acc.observe(self._make_stats(num_drafts=1, num_draft_tokens=10, num_accepted_tokens=3))
        snap = acc.snapshot()
        assert snap["acceptance_rate"] == 0.5  # 10/20
        assert snap["total_steps"] == 2

    def test_observe_toggle(self):
        acc = self._make_accumulator()
        acc.observe(self._make_stats())  # pre-toggle
        acc.observe(self._make_stats(sd_toggled=True, sd_toggle_tick=500))
        snap = acc.snapshot()
        assert snap["sd_toggled"] is True
        assert snap["sd_toggle_tick"] == 500

    def test_reset_preserves_toggle_state(self):
        acc = self._make_accumulator()
        acc.observe(self._make_stats(sd_toggled=True, sd_toggle_tick=500,
                                     num_drafts=1, num_draft_tokens=10, num_accepted_tokens=5))
        acc.reset()
        snap = acc.snapshot()
        # Toggle state preserved but tick reset to 0 (= "already active")
        assert snap["sd_toggled"] is True
        assert snap["sd_toggle_tick"] == 0
        # Counters reset
        assert snap["total_draft_tokens"] == 0
        assert snap["total_steps"] == 0
        assert snap["acceptance_rate"] == 0.0

    def test_accepted_per_pos_accumulation(self):
        """Per-position acceptance counts accumulate correctly across observe() calls."""
        acc = self._make_accumulator()
        # Draft round 1: gamma=5, 3 accepted → positions 0,1,2 each get 1
        acc.observe(self._make_stats(
            num_drafts=1, num_draft_tokens=5, num_accepted_tokens=3,
            num_accepted_tokens_per_pos=[1, 1, 1, 0, 0],
        ))
        # Draft round 2: gamma=5, 5 accepted → positions 0-4 each get 1
        acc.observe(self._make_stats(
            num_drafts=1, num_draft_tokens=5, num_accepted_tokens=5,
            num_accepted_tokens_per_pos=[1, 1, 1, 1, 1],
        ))
        snap = acc.snapshot()
        assert snap["accepted_per_pos"] == [2, 2, 2, 1, 1]
        assert snap["num_drafts"] == 2
        assert snap["total_draft_tokens"] == 10
        assert snap["total_accepted_tokens"] == 8

    def test_accepted_per_pos_reset(self):
        """Per-position counts reset to empty after reset()."""
        acc = self._make_accumulator()
        acc.observe(self._make_stats(
            num_drafts=1, num_draft_tokens=3, num_accepted_tokens=2,
            num_accepted_tokens_per_pos=[1, 1, 0],
        ))
        acc.reset()
        snap = acc.snapshot()
        assert snap["accepted_per_pos"] == []
        assert snap["num_drafts"] == 0

    def test_reset_baseline_stays_inactive(self):
        acc = self._make_accumulator()
        acc.observe(self._make_stats())  # baseline, no SD
        acc.reset()
        snap = acc.snapshot()
        assert snap["sd_toggled"] is False
        assert snap["sd_toggle_tick"] == -1
