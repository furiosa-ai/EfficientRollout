# SPDX-License-Identifier: Apache-2.0
"""KV selection for sparse attention drafting.

Phase 1b: Block-level selection via Q·K^T scoring.
Phase 2:  Token-level selection (removed, see .omc/plans/).

Strategy: recent_window tokens (always) + top-K important past tokens/blocks.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Phase 1b: Block-level scoring (original)
# ──────────────────────────────────────────────────────────────────────


def score_blocks_from_attn_probs(
    attn_probs: torch.Tensor,
    query_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Score KV blocks from captured attention probabilities.

    Uses softmax(Q·K^T) dumped by FA2 kernel — zero-cost since the
    attention computation already happened during verification.

    Args:
        attn_probs: [batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded]
            Attention probabilities from FA2 (post-softmax).
        query_indices: [num_queries] — indices into seqlen_q dimension
            to select (e.g., first_target + bonus positions).
        seq_lens: [batch_size]
        block_size: tokens per block

    Returns:
        block_scores: [batch_size, max_blocks] — importance per block.
    """
    batch_size = attn_probs.shape[0]
    num_heads = attn_probs.shape[1]
    seqlen_k = attn_probs.shape[3]
    device = attn_probs.device

    max_blocks = (seqlen_k + block_size - 1) // block_size

    # Select query positions and average across them and heads.
    # attn_probs[:, :, query_indices, :] → [batch, heads, num_q, seqlen_k]
    if query_indices is not None and query_indices.numel() > 0:
        selected = attn_probs[:, :, query_indices, :]
    else:
        selected = attn_probs

    # Average over heads and query positions → [batch, seqlen_k]
    avg_attn = selected.mean(dim=(1, 2))

    # Reshape into blocks and sum → [batch, max_blocks]
    # Pad seqlen_k to multiple of block_size
    pad_len = max_blocks * block_size - seqlen_k
    if pad_len > 0:
        avg_attn = torch.nn.functional.pad(avg_attn, (0, pad_len), value=0.0)

    block_scores = avg_attn.reshape(batch_size, max_blocks, block_size).sum(dim=-1)

    # Mask blocks beyond each request's actual block count
    num_blocks_per_req = (seq_lens + block_size - 1) // block_size
    block_arange = torch.arange(max_blocks, device=device).unsqueeze(0)
    mask = block_arange >= num_blocks_per_req.unsqueeze(1)
    block_scores.masked_fill_(mask, float("-inf"))

    return block_scores


def score_blocks(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score each KV block by attention importance (Q·K^T).

    Args:
        query: [batch_size, num_heads, head_dim] — query from draft step 0.
        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
        block_table: [batch_size, max_blocks_per_seq]
        seq_lens: [batch_size]
        block_size: tokens per block

    Returns:
        block_scores: [batch_size, max_blocks_per_seq] — importance per block.
            Unused blocks (beyond seq_len) are set to -inf.
    """
    key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]
    batch_size = block_table.shape[0]
    max_blocks = block_table.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    num_q_heads = query.shape[1]
    device = query.device

    # GQA: average query heads per KV head group
    if num_q_heads != num_kv_heads:
        heads_per_group = num_q_heads // num_kv_heads
        q = query.reshape(batch_size, num_kv_heads, heads_per_group, head_dim)
        q = q.mean(dim=2)  # [batch, num_kv_heads, head_dim]
    else:
        q = query  # [batch, num_kv_heads, head_dim]

    # Gather K blocks for all sequences: key_cache[block_table]
    # block_table: [batch, max_blocks] → indices into key_cache
    # Result: [batch, max_blocks, block_size, num_kv_heads, head_dim]
    K_blocks = key_cache[block_table]

    # Score: sum of Q·K^T over tokens in each block, averaged over heads.
    # q: [batch, num_kv_heads, head_dim]
    # K_blocks: [batch, max_blocks, block_size, num_kv_heads, head_dim]
    # → scores: [batch, max_blocks]
    scores = torch.einsum('bhd,bnshd->bn', q, K_blocks)
    # scores is summed over block_size and num_kv_heads.

    # Mask blocks beyond each request's actual block count
    num_blocks_per_req = (seq_lens + block_size - 1) // block_size  # [batch]
    block_arange = torch.arange(max_blocks, device=device).unsqueeze(0)
    mask = block_arange >= num_blocks_per_req.unsqueeze(1)
    scores.masked_fill_(mask, float("-inf"))

    if query_mask is not None:
        # Requests without valid query guidance should not contribute to top-k.
        scores.masked_fill_(~query_mask.unsqueeze(1), float("-inf"))

    return scores


def build_sparse_block_table(
    block_table: torch.Tensor,
    block_scores: torch.Tensor,
    seq_lens: torch.Tensor,
    num_topk_blocks: int,
    num_recent_blocks: int,
    block_size: int,
    candidate_cap_blocks: int = 0,
    query_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build sparse block_table: top-K important + recent blocks.

    Args:
        block_table: [batch_size, max_blocks_per_seq] — original
        block_scores: [batch_size, max_blocks_per_seq] — from score_blocks()
        seq_lens: [batch_size]
        num_topk_blocks: number of important past blocks to select
        num_recent_blocks: number of recent blocks (always included)
        block_size: tokens per block

    Returns:
        sparse_block_table: [batch_size, num_topk_blocks + num_recent_blocks]
        sparse_seq_lens: [batch_size] — token count used by FlashAttention
        sparse_capacity_tokens: [batch_size] — sparse table token capacity
        sparse_max_seqlen_k: host int upper bound for FlashAttention
    """
    batch_size = block_table.shape[0]
    max_blocks = block_table.shape[1]
    device = block_table.device

    num_blocks_per_req = (seq_lens + block_size - 1) // block_size

    # Identify recent blocks (last num_recent_blocks per request)
    # Mask recent blocks in scores so they don't compete for top-K
    scores_for_topk = block_scores.clone()
    # Vectorized recent block masking: for each request, mask blocks
    # from (num_blocks - num_recent_blocks) to num_blocks.
    block_arange = torch.arange(max_blocks, device=device).unsqueeze(0)
    recent_start_per_req = (
        num_blocks_per_req - num_recent_blocks).clamp(min=0).unsqueeze(1)
    recent_mask = (
        (block_arange >= recent_start_per_req)
        & (block_arange < num_blocks_per_req.unsqueeze(1)))
    scores_for_topk.masked_fill_(recent_mask, float('-inf'))

    # Optionally cap candidate past blocks to the latest N blocks before
    # recent-window blocks. This reduces top-k compute pressure.
    if candidate_cap_blocks > 0:
        past_end_per_req = (num_blocks_per_req - num_recent_blocks).clamp(min=0)
        cand_start_per_req = (past_end_per_req - candidate_cap_blocks).clamp(min=0)
        cap_mask = (
            (block_arange >= cand_start_per_req.unsqueeze(1))
            & (block_arange < past_end_per_req.unsqueeze(1))
        )
        scores_for_topk.masked_fill_(~cap_mask, float("-inf"))

    # Select top-K blocks from past (non-recent) blocks
    effective_k = min(num_topk_blocks, max_blocks - num_recent_blocks)
    if effective_k <= 0:
        effective_k = 0

    if effective_k > 0:
        _, topk_block_indices = scores_for_topk.topk(
            effective_k, dim=-1)  # [batch, effective_k]
    else:
        topk_block_indices = torch.zeros(
            batch_size, 0, dtype=torch.long, device=device)

    # Build recent block indices (vectorized, no Python loop)
    # recent_indices[i] = [recent_start_i, recent_start_i+1, ..., recent_start_i+num_recent-1]
    if num_recent_blocks > 0:
        offsets = torch.arange(
            num_recent_blocks, device=device, dtype=torch.long).unsqueeze(0)
        recent_indices = (recent_start_per_req.long() + offsets).clamp(
            max=max_blocks - 1)  # [batch, num_recent]
    else:
        recent_indices = torch.zeros(
            batch_size, 0, device=device, dtype=torch.long)

    if query_mask is not None and effective_k > 0:
        empty_topk = torch.zeros_like(topk_block_indices)
        topk_block_indices = torch.where(
            query_mask.unsqueeze(1),
            topk_block_indices,
            empty_topk,
        )

    # Combine: [topk_blocks | recent_blocks]
    if effective_k > 0:
        all_logical_indices = torch.cat(
            [topk_block_indices, recent_indices], dim=1)
    else:
        all_logical_indices = recent_indices

    # Clamp to valid block range per request (topk may select -inf positions)
    max_valid = (num_blocks_per_req - 1).clamp(min=0).unsqueeze(1)
    all_logical_indices = all_logical_indices.clamp(min=0)
    all_logical_indices = torch.min(
        all_logical_indices, max_valid.expand_as(all_logical_indices))

    # Sort ascending for cache-friendly reads, then map to physical blocks
    all_logical_indices, _ = all_logical_indices.sort(dim=1)
    sparse_block_table = block_table.gather(1, all_logical_indices)

    # Sparse seq_lens: number of tokens covered by sparse blocks.
    # FlashAttention reads ceil(seqused_k / block_size) blocks from
    # block_table. Set to total_sparse_blocks * block_size so all
    # blocks in sparse_block_table are read. Clamped to original
    # seq_lens to avoid exceeding actual KV content.
    total_sparse_blocks = int(all_logical_indices.shape[1])
    sparse_capacity_tokens = torch.full(
        (batch_size,), total_sparse_blocks * block_size,
        device=device, dtype=torch.int32)
    sparse_seq_lens = torch.min(sparse_capacity_tokens, seq_lens.int())
    sparse_max_seqlen_k = total_sparse_blocks * block_size

    return (
        sparse_block_table,
        sparse_seq_lens,
        sparse_capacity_tokens,
        sparse_max_seqlen_k,
    )
