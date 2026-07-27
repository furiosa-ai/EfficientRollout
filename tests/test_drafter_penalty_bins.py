"""Differential test: incremental penalty bin counts == the upstream rebuild.

The drafter's penalties path (quant_self_proposer._apply_logits_processors_binned)
replaces the per-draft-step rebuild of a padded [B, L] token-history tensor with
an incrementally-updated histogram. This test asserts the two produce
bit-identical logits over a full gamma-step draft cycle, including the
async-scheduling -1 placeholder case and ragged history lengths.

The two methods are exercised UNBOUND on the shipped class rather than retyped
here, so the test guards the real implementation: neither method touches `self`,
so passing None for it is sufficient and keeps the test free of engine setup.

Run:  python tests/test_drafter_penalty_bins.py
"""

import sys
import types

import torch

sys.path.insert(0, "third_party/vllm")

from vllm.v1.sample.ops.penalties import apply_all_penalties  # noqa: E402
from vllm.v1.spec_decode.quant_self_proposer import (  # noqa: E402
    QuantizedSelfDrafterProposer as _Proposer,
)

# The two methods under test, called unbound against the shipped code.
_init_bins = _Proposer._init_penalty_bins
_apply_binned = _Proposer._apply_logits_processors_binned


def _fake_metadata(prompt_t, history, pres, freq, rep):
    """Minimal SamplingMetadata stand-in: only the fields the binned path reads."""
    return types.SimpleNamespace(
        prompt_token_ids=prompt_t,
        output_token_ids=history,
        presence_penalties=pres,
        frequency_penalties=freq,
        repetition_penalties=rep,
        allowed_token_ids_mask=None,
        logitsprocs=types.SimpleNamespace(non_argmax_invariant=()),
    )


def upstream(logits, prompt_t, history, pres, freq, rep):
    """What the drafter did at each draft step before this change."""
    return apply_all_penalties(
        logits.clone(), prompt_t, pres, freq, rep, history
    )


def run_case(name, B, vocab, prompt_len, hist_lens, gamma, device,
             with_placeholders=False, rep_val=1.0, pres_val=0.0,
             freq_val=0.05, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    prompt_t = torch.randint(
        0, vocab, (B, prompt_len), generator=g, dtype=torch.int64
    ).to(device)
    history = [
        torch.randint(0, vocab, (n,), generator=g).tolist()
        for n in hist_lens
    ]
    if with_placeholders:
        # async-scheduling rows carrying -1 placeholders
        history[0] = [-1] * len(history[0])
    drafts = torch.randint(
        0, vocab, (gamma, B), generator=g, dtype=torch.int64
    ).to(device)

    freq = torch.full((B,), freq_val, dtype=torch.float32, device=device)
    pres = torch.full((B,), pres_val, dtype=torch.float32, device=device)
    rep = torch.full((B,), rep_val, dtype=torch.float32, device=device)

    # --- state init: the shipped _init_penalty_bins, called unbound ---
    meta = _fake_metadata(prompt_t, history, pres, freq, rep)
    bins = _init_bins(None, meta, B, vocab)

    live_history = [list(h) for h in history]
    for step in range(gamma):
        logits = torch.randn(
            B, vocab, generator=g, dtype=torch.float32
        ).to(device)
        ref = upstream(logits, prompt_t, live_history, pres, freq, rep)
        got = _apply_binned(None, logits.clone(), meta, bins)
        if not torch.equal(ref, got):
            d = (ref - got).abs().max().item()
            raise AssertionError(
                f"{name}: mismatch at draft step {step} (max|diff|={d:.3e})"
            )
        # advance both representations by one drafted token per request,
        # exactly as propose() does
        tok = drafts[step]
        bins["counts"].scatter_add_(1, tok.long().unsqueeze(1), bins["ones"])
        for i, t in enumerate(tok.tolist()):
            live_history[i].append(t)
    print(f"  PASS  {name}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 2048  # small vocab keeps the [B, vocab] compares fast
    print(f"binned-penalty differential test (device={device}, vocab={vocab})")
    run_case("freq only, uniform history", 8, vocab, 64,
             [500] * 8, 9, device, seed=1)
    run_case("freq only, ragged history", 8, vocab, 64,
             [0, 1, 37, 500, 1200, 3, 900, 250], 9, device, seed=2)
    run_case("freq+presence+repetition", 6, vocab, 32,
             [400, 401, 7, 0, 1000, 55], 11, device,
             rep_val=1.15, pres_val=0.2, seed=3)
    run_case("async -1 placeholders", 4, vocab, 16,
             [300, 300, 300, 300], 5, device,
             with_placeholders=True, seed=4)
    run_case("negative frequency penalty", 4, vocab, 16,
             [120, 60, 0, 900], 7, device, freq_val=-0.3, seed=5)
    print("all cases bit-identical")


if __name__ == "__main__":
    main()
