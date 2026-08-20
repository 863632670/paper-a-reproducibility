"""
A1: S13 G6-null d=188 sign-flip reproducibility test.

Narrow, deliberate scope per your instruction -- do NOT re-verify all of
106. Answer only: is the d=188 Delta(hist-G6) sign flip for S13 a real
directional effect, or stochastic fluctuation?

Kept IDENTICAL to paper108_phase2b_grammar_null.py / _repeat2.py's own
pipeline (same S13, same G6-null construction, same measurement
pipeline: W=100, N_PEAKS=15, N_ACCESS_TRIALS=5, ACCESS_PERTURB=5,
ACCESS_ITER=80, RANDOM_PEAK_ITER=120, same RNG-seed-construction
formula). Only change: scope narrowed to S13 x d=188 x G6 only (drop
S18/S19, d=0/100, G5) -- the cheapest possible test of just this one
cell -- and repeat count raised from n=2 (already existing) to n=10
total by adding 8 new independent repeats (repeat 3-10), each with its
own SEED_OFFSET, non-overlapping with repeat1 (offset=0) and repeat2
(offset=60000).

Existing results (already computed, reused not re-run):
  repeat 1 (offset=0):     Delta(hist-G6) = +0.277
  repeat 2 (offset=60000): Delta(hist-G6) = -0.089
"""
import numpy as np
import common106 as c

W = 100
N_PEAKS = 15
N_ACCESS_TRIALS = 5
ACCESS_PERTURB = 5
ACCESS_ITER = 80
RANDOM_PEAK_ITER = 120
D = 188
NEW_OFFSETS = [120000, 180000, 240000, 300000, 360000, 420000, 480000, 540000]


def chain_to_source(blocks, chain):
    s_bin = np.concatenate([blocks[i] for i in chain])
    return np.where(s_bin == 1, 1.0, -1.0)


def hill_climb_on_chain(blocks, chain, macro_trunc, rng, n_iter):
    current_chain = list(chain)
    current_o = c.O_of_B(chain_to_source(blocks, current_chain), macro_trunc, B=500, rng_seed=0)
    n_b = len(blocks)
    for _ in range(n_iter):
        i, j = rng.choice(n_b, size=2, replace=False)
        candidate = list(current_chain)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cand_o = c.O_of_B(chain_to_source(blocks, candidate), macro_trunc, B=500, rng_seed=0)
        if cand_o > current_o:
            current_chain = candidate
            current_o = cand_o
    return current_chain, current_o


def accessibility(blocks, chain, macro_trunc, rng):
    vals = []
    n_b = len(blocks)
    for _ in range(N_ACCESS_TRIALS):
        perturbed = list(chain)
        for _ in range(ACCESS_PERTURB):
            i, j = rng.choice(n_b, size=2, replace=False)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        _, final_o = hill_climb_on_chain(blocks, perturbed, macro_trunc, rng, n_iter=ACCESS_ITER)
        vals.append(final_o)
    return np.mean(vals)


def block_aware_distance(chain, identity):
    return sum(1 for a, b in zip(chain, identity) if a != b)


def perturb_to_exact_distance(identity_chain, target_d, rng, n_blocks, max_attempts=3000):
    if target_d == 0:
        return list(identity_chain), 0
    chain = list(identity_chain)
    for attempt in range(max_attempts):
        current_d = block_aware_distance(chain, identity_chain)
        if current_d == target_d:
            return chain, current_d
        i, j = rng.choice(n_blocks, size=2, replace=False)
        candidate = list(chain)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cand_d = block_aware_distance(candidate, identity_chain)
        if abs(cand_d - target_d) <= abs(current_d - target_d):
            chain = candidate
    return chain, block_aware_distance(chain, identity_chain)


def block_eulerian_chain_order_k(source, w, order, rng_seed):
    n = len(source)
    n_blocks = n // w
    s_bin = (source[:n_blocks * w] > 0).astype(int)
    blocks = [s_bin[i * w:(i + 1) * w] for i in range(n_blocks)]
    entry_ctx = []
    for i in range(n_blocks):
        start = i * w
        if start >= order:
            entry_ctx.append(tuple(s_bin[start - order:start]))
        else:
            entry_ctx.append(None)
    exit_ctx = [tuple(b[-order:]) for b in blocks]
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


d13 = np.load("cache/s13_m13.npz")
source, macro = d13["source13"], d13["macro13"]
n_blocks = len(source) // W
macro_trunc = macro[:n_blocks * W]
blocks_true = [(source[i * W:(i + 1) * W] > 0).astype(int) for i in range(n_blocks)]
identity_chain = list(range(n_blocks))

deltas = {0: 0.277, 60000: -0.089}  # repeat 1, repeat 2 (already computed, reused)

for offset in NEW_OFFSETS:
    hist_access = []
    for k in range(1, N_PEAKS + 1):
        rng_gen = np.random.RandomState(offset + 30000 + D * 1000 + k)
        chain, _ = perturb_to_exact_distance(identity_chain, D, rng_gen, n_blocks)
        rng_climb = np.random.RandomState(offset + 30000 + D * 1000 + k + 5000)
        climbed, _ = hill_climb_on_chain(blocks_true, chain, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
        rng_acc = np.random.RandomState(offset + 30000 + D * 1000 + k + 10000)
        hist_access.append(accessibility(blocks_true, climbed, macro_trunc, rng_acc))
    hist_access = np.array(hist_access)

    g6_access = []
    for k in range(1, N_PEAKS + 1):
        g6_blocks, g6_chain0, _ = block_eulerian_chain_order_k(source, W, order=6, rng_seed=offset + 32000 + D * 1000 + k)
        rng_pert = np.random.RandomState(offset + 32000 + D * 1000 + k + 2000)
        g6_chain, _ = perturb_to_exact_distance(g6_chain0, D, rng_pert, n_blocks)
        rng_climb = np.random.RandomState(offset + 32000 + D * 1000 + k + 5000)
        climbed, _ = hill_climb_on_chain(g6_blocks, g6_chain, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
        rng_acc = np.random.RandomState(offset + 32000 + D * 1000 + k + 10000)
        g6_access.append(accessibility(g6_blocks, climbed, macro_trunc, rng_acc))
    g6_access = np.array(g6_access)

    delta = hist_access.mean() - g6_access.mean()
    deltas[offset] = delta
    print(f"offset={offset:6d}: hist_mean={hist_access.mean():.3f} g6_mean={g6_access.mean():.3f} delta={delta:+.3f}")

print("\n=== All 10 repeats, S13, d=188, Delta(hist-G6) ===")
all_deltas = list(deltas.values())
for i, (offset, d) in enumerate(deltas.items(), 1):
    print(f"repeat {i} (offset={offset}): {d:+.3f}")

n_pos = sum(1 for d in all_deltas if d > 0)
n_neg = sum(1 for d in all_deltas if d < 0)
print(f"\nSign count: {n_pos} positive, {n_neg} negative, out of {len(all_deltas)}")
print(f"Mean delta: {np.mean(all_deltas):+.3f}, std: {np.std(all_deltas):.3f}")

from scipy import stats
t, p = stats.ttest_1samp(all_deltas, 0)
print(f"One-sample t-test vs 0: t={t:.3f} p={p:.3f}")

np.savez("cache/A1_S13_G6_d188_repeats.npz", deltas=np.array(all_deltas), offsets=np.array(list(deltas.keys())))
