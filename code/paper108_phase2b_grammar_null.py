"""
108 Connectivity Redesign, Phase 2B: G5/G6 grammar-null comparison, per
your Phase 2B design. Tests whether Phase 1's historical-neighborhood
advantage is just "preserving a matched grammar order" (which 107-2/3
already showed insufficient for reproducing history-level compatibility
via generative search) or something beyond grammar-order complexity.

Compares, at the SAME distance-0 anchor point (i.e., not perturbed --
this tests the PEAKS themselves, matching 107's comparison against
historical-level compatibility directly) and additionally at d=188
(matching Phase 2A's second distance, so a direct read-across is
possible): true history vs Null-G5 (order-5 grammar-preserving Eulerian
reconstruction) vs Null-G6 (order-6) vs Null-random (order-0, already in
Phase 1/2A for reference).

Metric: A_history - A_G5, A_history - A_G6 (accessibility difference).
"""
import numpy as np
import common106 as c

W = 100
N_PEAKS = 15   # matches Phase 1's N_PER_D for comparability
N_ACCESS_TRIALS = 5
ACCESS_PERTURB = 5
ACCESS_ITER = 80
RANDOM_PEAK_ITER = 120
D_GRID = [0, 188]


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
    """Reused verbatim from paper108_expA3_jackpot_hypothesis.py."""
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


objects = {"S13": 180013, "S18": 180018, "S19": 180019}
trajs = {}
d13 = np.load("cache/s13_m13.npz")
trajs[180013] = (d13["source13"], d13["macro13"])
for name, seed in objects.items():
    if seed != 180013:
        d = np.load(f"cache/s{seed - 180000:02d}_traj.npz")
        trajs[seed] = (d["source"], d["macro"])

results = {}

for name, seed in objects.items():
    source, macro = trajs[seed]
    n_blocks = len(source) // W
    macro_trunc = macro[:n_blocks * W]
    blocks_true = [(source[i * W:(i + 1) * W] > 0).astype(int) for i in range(n_blocks)]
    identity_chain = list(range(n_blocks))

    print(f"\n{'='*70}\n{name} (seed {seed})\n{'='*70}")

    obj_results = {}
    for d in D_GRID:
        print(f"\n  d_B={d}:")

        # historical, at distance d (block-preserving, matching Phase 1's Mode 1)
        hist_access = []
        for k in range(1, N_PEAKS + 1):
            rng_gen = np.random.RandomState(30000 + d * 1000 + k)
            chain, _ = perturb_to_exact_distance(identity_chain, d, rng_gen, n_blocks)
            rng_climb = np.random.RandomState(30000 + d * 1000 + k + 5000)
            climbed, _ = hill_climb_on_chain(blocks_true, chain, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
            rng_acc = np.random.RandomState(30000 + d * 1000 + k + 10000)
            hist_access.append(accessibility(blocks_true, climbed, macro_trunc, rng_acc))
        hist_access = np.array(hist_access)

        # Null-G5: order-5 grammar-preserving Eulerian reconstruction, then
        # perturbed to the SAME block-aware distance d from ITS OWN identity
        # (i.e. d block-swaps applied to the G5 reconstruction itself)
        g5_access = []
        for k in range(1, N_PEAKS + 1):
            rng_g5 = np.random.RandomState(31000 + d * 1000 + k)
            g5_blocks, g5_chain0, _ = block_eulerian_chain_order_k(source, W, order=5, rng_seed=31000 + d * 1000 + k)
            g5_identity = list(range(len(g5_blocks)))
            rng_pert = np.random.RandomState(31000 + d * 1000 + k + 2000)
            g5_chain, _ = perturb_to_exact_distance(g5_chain0, d, rng_pert, n_blocks) if d > 0 else (g5_chain0, 0)
            rng_climb = np.random.RandomState(31000 + d * 1000 + k + 5000)
            climbed, _ = hill_climb_on_chain(g5_blocks, g5_chain, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
            rng_acc = np.random.RandomState(31000 + d * 1000 + k + 10000)
            g5_access.append(accessibility(g5_blocks, climbed, macro_trunc, rng_acc))
        g5_access = np.array(g5_access)

        # Null-G6: order-6
        g6_access = []
        for k in range(1, N_PEAKS + 1):
            rng_g6 = np.random.RandomState(32000 + d * 1000 + k)
            g6_blocks, g6_chain0, _ = block_eulerian_chain_order_k(source, W, order=6, rng_seed=32000 + d * 1000 + k)
            rng_pert = np.random.RandomState(32000 + d * 1000 + k + 2000)
            g6_chain, _ = perturb_to_exact_distance(g6_chain0, d, rng_pert, n_blocks) if d > 0 else (g6_chain0, 0)
            rng_climb = np.random.RandomState(32000 + d * 1000 + k + 5000)
            climbed, _ = hill_climb_on_chain(g6_blocks, g6_chain, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
            rng_acc = np.random.RandomState(32000 + d * 1000 + k + 10000)
            g6_access.append(accessibility(g6_blocks, climbed, macro_trunc, rng_acc))
        g6_access = np.array(g6_access)

        delta_g5 = hist_access.mean() - g5_access.mean()
        delta_g6 = hist_access.mean() - g6_access.mean()
        print(f"    History access: mean={hist_access.mean():.3f}")
        print(f"    Null-G5 access: mean={g5_access.mean():.3f}   Delta(hist-G5)={delta_g5:+.3f}")
        print(f"    Null-G6 access: mean={g6_access.mean():.3f}   Delta(hist-G6)={delta_g6:+.3f}")
        obj_results[d] = dict(hist=hist_access, g5=g5_access, g6=g6_access,
                               delta_g5=delta_g5, delta_g6=delta_g6)

    results[name] = obj_results

print("\n" + "=" * 70)
print("CROSS-OBJECT SUMMARY")
print("=" * 70)
for label, key in [("Delta(hist - G5)", "delta_g5"), ("Delta(hist - G6)", "delta_g6")]:
    print(f"\n--- {label} ---")
    print(f"{'d':>6}" + "".join(f"{name:>12}" for name in objects))
    for d in D_GRID:
        row = [results[name][d][key] for name in objects]
        print(f"{d:>6}" + "".join(f"{v:>+12.3f}" for v in row))

np.savez("cache/phase2b_results.npz", **{
    f"{name}_{d}_{k}": v for name in objects for d in D_GRID
    for k, v in results[name][d].items() if isinstance(v, np.ndarray)
})
