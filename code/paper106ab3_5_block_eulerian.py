"""
Reconstruction of Paper 106-AB3-5 Groups A/B (block-Eulerian reordering causal test).

CATEGORY C: reimplemented from papers/Paper106AB3_5_Strong_Support_w100.docx.

Method (as described): true S13 divided into non-overlapping blocks of size w.
Each block has an entry context (5-symbol history immediately preceding the
block in the ORIGINAL sequence) and exit context (block's own final 5
symbols). Randomized greedy chaining starting from block 0: at each step,
among still-unused blocks, pick a random one whose entry context matches the
current chain's exit context (G5-consistent junction); if none match, pick a
random remaining block. Concatenate chosen blocks; pair with truncated true
M13; O500 = O_of_B(reordered_source, macro_truncated, B=500, rng_seed=0).

Paper's reported table (O500, change from baseline 1.993):
  w=50:  seed1=0.261(-1.732) seed2=3.090(+1.096) seed3=0.767(-1.227)
  w=100: seed1=0.583(-1.410) seed2=0.454(-1.539) seed3=-0.251(-2.245)
"""
import numpy as np
import common106 as c

GRAMMAR_ORDER = 5


def block_eulerian_chain(source, w, rng_seed):
    """Same algorithm as block_eulerian_reorder, but returns (blocks, chain)
    instead of the concatenated array, so callers (e.g. AB3-C2's simulated
    annealing) can continue optimizing the block ORDER directly."""
    n = len(source)
    n_blocks = n // w
    s_bin = (source[:n_blocks * w] > 0).astype(int)
    blocks = [s_bin[i * w:(i + 1) * w] for i in range(n_blocks)]

    entry_ctx = []
    for i in range(n_blocks):
        start = i * w
        if start >= GRAMMAR_ORDER:
            entry_ctx.append(tuple(s_bin[start - GRAMMAR_ORDER:start]))
        else:
            entry_ctx.append(None)
    exit_ctx = [tuple(b[-GRAMMAR_ORDER:]) for b in blocks]

    rng = np.random.RandomState(rng_seed)
    used = [False] * n_blocks
    chain = [0]
    used[0] = True
    current_exit = exit_ctx[0]

    for _ in range(n_blocks - 1):
        candidates = [i for i in range(n_blocks) if not used[i] and entry_ctx[i] == current_exit]
        if candidates:
            nxt = candidates[rng.randint(len(candidates))]
        else:
            remaining = [i for i in range(n_blocks) if not used[i]]
            nxt = remaining[rng.randint(len(remaining))]
        chain.append(nxt)
        used[nxt] = True
        current_exit = exit_ctx[nxt]

    return blocks, chain, rng


def block_eulerian_reorder(source, w, rng_seed):
    n = len(source)
    n_blocks = n // w
    s_bin = (source[:n_blocks * w] > 0).astype(int)
    blocks = [s_bin[i * w:(i + 1) * w] for i in range(n_blocks)]

    # entry context = 5 symbols immediately preceding block i in the ORIGINAL sequence
    # (for block 0, there is no preceding context; treat as None / always-matchable only by itself)
    entry_ctx = []
    for i in range(n_blocks):
        start = i * w
        if start >= GRAMMAR_ORDER:
            entry_ctx.append(tuple(s_bin[start - GRAMMAR_ORDER:start]))
        else:
            entry_ctx.append(None)
    exit_ctx = [tuple(b[-GRAMMAR_ORDER:]) for b in blocks]

    rng = np.random.RandomState(rng_seed)
    used = [False] * n_blocks
    chain = [0]
    used[0] = True
    current_exit = exit_ctx[0]

    for _ in range(n_blocks - 1):
        candidates = [i for i in range(n_blocks) if not used[i] and entry_ctx[i] == current_exit]
        if candidates:
            nxt = candidates[rng.randint(len(candidates))]
        else:
            remaining = [i for i in range(n_blocks) if not used[i]]
            nxt = remaining[rng.randint(len(remaining))]
        chain.append(nxt)
        used[nxt] = True
        current_exit = exit_ctx[nxt]

    new_bin = np.concatenate([blocks[i] for i in chain])
    return np.where(new_bin == 1, 1.0, -1.0)


data = np.load("cache/s13_m13.npz")
source13 = data["source13"]
macro13 = data["macro13"]

baseline = c.O_of_B(source13, macro13, B=500, rng_seed=0)
print("baseline O500:", baseline, "(paper: 1.993)")
print()

paper_vals = {
    50: [0.261, 3.090, 0.767],
    100: [0.583, 0.454, -0.251],
}

for w in [50, 100]:
    print(f"w={w}")
    n_blocks = len(source13) // w
    macro_trunc = macro13[:n_blocks * w]
    for i, seed in enumerate([1, 2, 3]):
        reordered = block_eulerian_reorder(source13, w, rng_seed=seed)
        o = c.O_of_B(reordered, macro_trunc, B=500, rng_seed=0)
        change = o - baseline
        print(f"  seed={seed}: O500={o:.3f} (change {change:+.3f})   paper: {paper_vals[w][i]:.3f}")
    print()
