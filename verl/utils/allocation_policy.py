"""Pluggable prompt-to-GPU allocation policies for rollout.

Replaces the hardcoded least-requests load balancing in AsyncLLMServerManager
with configurable allocation strategies that can account for expected response
lengths and long-tail risk.

Terminology:
    - "prompt" = a unique input. Each prompt produces G rollouts.
    - All G rollouts of a prompt are assigned to the same GPU (for prefix caching).
    - The allocate() interface receives unique prompts, NOT expanded requests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PromptInfo:
    """Per-prompt metadata used by allocation policies.

    Each instance represents a unique prompt (NOT an individual rollout).
    All G rollouts of a prompt are implicitly assigned to the same GPU.
    """

    key: int  # dataset index or hash
    prompt_length: int  # number of prompt tokens
    mu: Optional[float] = None  # expected response length (from history)
    tail_prob: float = 0.0  # long-tail probability (EMA of fraction of rollouts > κ*median)


class AllocationPolicy(ABC):
    """Base class for prompt-to-GPU allocation policies."""

    @abstractmethod
    def allocate(
        self, prompts: list[PromptInfo], num_gpus: int
    ) -> dict[int, list[int]]:
        """Assign unique prompts to GPUs.

        All G rollouts of a prompt are implicitly assigned to the same GPU
        (enables prefix caching).

        Args:
            prompts: list of PromptInfo, one per unique prompt.
            num_gpus: number of vLLM server replicas (GPUs).

        Returns:
            Dict mapping gpu_idx -> list of prompt indices
            (into the prompts list).
        """
        ...


class DefaultPolicy(AllocationPolicy):
    """No-op allocation: preserves original veRL least-requests LB behavior.

    Returns empty assignment, signaling that no explicit GPU pinning is used.
    Each request is dispatched by the existing min-heap load balancer in
    AsyncLLMServerManager._choose_server(), which is request-level and
    query-agnostic (same as pre-Phase-0 behavior).
    """

    def allocate(
        self, prompts: list[PromptInfo], num_gpus: int
    ) -> dict[int, list[int]]:
        return {}


class LengthBalancingPolicy(AllocationPolicy):
    """Balance by expected total response length (longest-first greedy).

    Balances total expected generation work across GPUs while keeping
    all rollouts of each prompt on the same GPU for prefix caching.

    TBD: requires wiring length history from RolloutLengthTracker into
    the allocation call chain (Phase 2).
    """

    def allocate(
        self, prompts: list[PromptInfo], num_gpus: int
    ) -> dict[int, list[int]]:
        raise NotImplementedError(
            "LengthBalancingPolicy.allocate() is not yet implemented. "
            "Requires Phase 2 integration: length history → allocation chain."
        )


class EffectiveLoadPolicy(AllocationPolicy):
    """Balance by effective load w_q that accounts for tail risk.

    This is the allocation policy:
        w_q = G * [(1 - p_q) * μ_q + p_q * L_q^tail]

    where p_q is the EMA long-tail probability (fraction of rollouts > κ*median),
    and L_q^tail = min(κ * μ_q, L_max).

    TBD: requires wiring length history from RolloutLengthTracker
    into the allocation call chain (Phase 2).

    Args:
        G: number of rollouts per prompt
        tail_amplification: κ factor for tail length amplification
        l_max: maximum response length
    """

    def __init__(
        self,
        G: int,
        tail_amplification: float = 2.0,
        l_max: int = 8192,
        **kwargs,
    ):
        self.G = G
        self.kappa = tail_amplification
        self.l_max = l_max

    def _compute_effective_load(self, p: PromptInfo) -> float:
        """Compute w_q for a single prompt."""
        mu = p.mu or 0.0
        tp = p.tail_prob
        if tp > 0:
            l_tail = min(self.kappa * mu, self.l_max)
            return self.G * ((1 - tp) * mu + tp * l_tail)
        return self.G * mu

    def allocate(
        self, prompts: list[PromptInfo], num_gpus: int
    ) -> dict[int, list[int]]:
        raise NotImplementedError(
            "EffectiveLoadPolicy.allocate() is not yet implemented. "
            "Requires Phase 2 integration: length history → allocation chain."
        )


def create_allocation_policy(
    policy_name: str,
    G: int = 1,
    l_max: int = 8192,
    tail_amplification: float = 2.0,
    **kwargs,
) -> AllocationPolicy:
    """Factory function to create an allocation policy from config.

    Args:
        policy_name: one of "default", "length_balancing", "effective_load"
        G: number of rollouts per prompt (for effective_load)
        l_max: maximum response length (for effective_load)
        tail_amplification: κ factor (for effective_load)

    Returns:
        An AllocationPolicy instance.
    """
    if policy_name == "default":
        return DefaultPolicy()
    elif policy_name == "length_balancing":
        return LengthBalancingPolicy()
    elif policy_name == "effective_load":
        return EffectiveLoadPolicy(
            G=G,
            tail_amplification=tail_amplification,
            l_max=l_max,
        )
    else:
        raise ValueError(
            f"Unknown allocation policy: {policy_name}. "
            f"Choose from: default, length_balancing, effective_load"
        )
