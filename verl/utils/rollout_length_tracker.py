"""Prompt-level response length history for allocation.

Collects (prompt_key, response_length) per rollout step, aggregates per epoch,
and provides per-prompt statistics including long-tail probability via EMA.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PromptStats:
    """Aggregated statistics for a single prompt across its rollouts."""

    mu: float  # mean response length
    max_len: int  # max response length observed
    std: float  # std of response lengths
    median: float  # median response length
    tail_prob: float = 0.0  # EMA of long-tail probability: fraction of rollouts > 2*median


class RolloutLengthTracker:
    """Tracks per-prompt response lengths across rollout steps and epochs.

    Long-tail definition:
        A rollout is "long-tail" if its response length > 2 * median(prompt's lengths).
        Per-prompt tail_prob = count(long-tail) / G, updated via EMA across epochs.

    Usage:
        tracker = RolloutLengthTracker(l_max=8192, ema_alpha=0.3)

        # After each rollout step:
        tracker.record_step(prompt_keys, response_lengths, epoch=current_epoch)

        # After each epoch completes:
        tracker.aggregate_epoch(epoch)

        # Query for allocation:
        stats = tracker.get_prompt_stats(prompt_key)
        all_stats = tracker.get_all_stats()
    """

    def __init__(self, l_max: int, ema_alpha: float = 0.3, tail_amplification: float = 2.0, **kwargs):
        """
        Args:
            l_max: maximum response length (from rollout config)
            ema_alpha: EMA smoothing factor for tail_prob (higher = more weight on recent)
            tail_amplification: κ factor — a rollout is "long-tail" if length > κ * median
        """
        self.l_max = l_max
        self.ema_alpha = ema_alpha
        self.kappa = tail_amplification
        # epoch -> {prompt_key -> [response_lengths]}
        self._history: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        # Aggregated stats from completed epochs (prompt_key -> PromptStats)
        self._aggregated: dict[int, PromptStats] = {}
        # Track which epochs have been aggregated
        self._aggregated_epochs: set[int] = set()

    def record_step(
        self,
        prompt_keys: np.ndarray,
        response_lengths: np.ndarray,
        epoch: int = 0,
    ) -> None:
        """Record response lengths from one rollout step.

        Args:
            prompt_keys: array of prompt identifiers (dataset indices or hashes)
            response_lengths: array of response lengths (number of generated tokens)
            epoch: current epoch number
        """
        epoch_history = self._history[epoch]
        for key, length in zip(prompt_keys, response_lengths):
            epoch_history[int(key)].append(int(length))

    def aggregate_epoch(self, epoch: int) -> None:
        """Compute per-prompt stats for a completed epoch.

        For each prompt, computes mu/max/std/median from this epoch's data,
        and updates tail_prob via EMA: p_new = α * p_epoch + (1-α) * p_old.

        Args:
            epoch: epoch number to aggregate
        """
        if epoch not in self._history:
            return
        self._aggregated_epochs.add(epoch)

        for key, lengths in self._history[epoch].items():
            arr = np.array(lengths, dtype=np.float64)
            med = float(np.median(arr))
            n_tail = int(np.sum(arr > self.kappa * med))
            p_epoch = n_tail / len(arr)

            prev = self._aggregated.get(key)
            if prev is not None:
                p_ema = self.ema_alpha * p_epoch + (1 - self.ema_alpha) * prev.tail_prob
            else:
                p_ema = p_epoch

            self._aggregated[key] = PromptStats(
                mu=float(arr.mean()),
                max_len=int(arr.max()),
                std=float(arr.std()) if len(arr) > 1 else 0.0,
                median=med,
                tail_prob=p_ema,
            )

        # Prune old raw history to bound memory (keep last 2 epochs)
        epochs_to_prune = [e for e in self._history if e < epoch - 1]
        for e in epochs_to_prune:
            del self._history[e]

    def get_prompt_stats(self, prompt_key: int) -> Optional[PromptStats]:
        """Get aggregated stats for a single prompt.

        Returns None if no history exists for this prompt.
        """
        return self._aggregated.get(prompt_key)

    def get_all_stats(self) -> dict[int, PromptStats]:
        """Get aggregated stats for all known prompts."""
        return dict(self._aggregated)

    def has_history(self) -> bool:
        """Whether any epoch has been aggregated (i.e., not cold start)."""
        return len(self._aggregated) > 0

    def get_tail_rate(self) -> float:
        """Mean tail_prob across all tracked prompts."""
        if not self._aggregated:
            return 0.0
        return sum(s.tail_prob for s in self._aggregated.values()) / len(self._aggregated)
