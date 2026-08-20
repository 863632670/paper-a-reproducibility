"""
108 Connectivity Redesign, Phase 1: full distance-grid run.
Per Paper108_Connectivity_Experiment_Protocol_Frozen.md Sections 3-4,
Amendments 1-2.

Scope: S13, S18, S19 (S19 run from scratch, per Paper108_S19_Provenance_Audit.md);
d in {0, 10, 25, 50, 100, 188, 376}; N=15 independent d-perturbed points
per (object,d); Mode 1 (block-preserving) only; symmetric optimization
treatment (both historical-at-d and Null-random peaks hill-climbed
120 iterations before accessibility is measured, per Amendment 1);
BOTH raw_climbed_height and accessibility tracked per Amendment 2;
Null-random ensemble is N=6 (matching 108-1.10C's original ensemble size).

This is a single full pass. A second independent repeat (for cross-repeat
ranking stability, per the frozen protocol Section 7) is a separate,
later run, not included here.
"""
import numpy as np
import common106 as c
from paper106ab3_5_block_eulerian import block_eulerian_chain

W = 100
N_PER_D = 15
N_NULL_PEAKS = 6
N_ACCESS_TRIALS = 5
ACCESS_PERTURB = 5
ACCESS_ITER = 80
RANDOM_PEAK_ITER = 120
D_GRID = [0, 10, 25, 50, 100, 188, 376]


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
    final_d = block_aware_distance(chain, identity_chain)
    if final_d != target_d:
        print(f"    WARNING: target d={target_d} not exactly reached, landed at {final_d}")
    return chain, final_d


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
    blocks = [(source[i * W:(i + 1) * W] > 0).astype(int) for i in range(n_blocks)]
    identity_chain = list(range(n_blocks))

    print(f"\n{'='*70}\n{name} (seed {seed}), n_blocks={n_blocks}\n{'='*70}")

    # Null-random ensemble: N=6 G5-Eulerian random peaks
    print(f"Null-random ensemble ({N_NULL_PEAKS} G5-Eulerian random peaks):")
    null_heights, null_accessibilities = [], []
    for k in range(1, N_NULL_PEAKS + 1):
        rng_gen = np.random.RandomState(9000 + k)
        eblocks, echain, _ = block_eulerian_chain(source, W, rng_seed=9000 + k)
        rng_climb = np.random.RandomState(9100 + k)
        peak_chain, peak_o = hill_climb_on_chain(eblocks, echain, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
        rng_access = np.random.RandomState(9200 + k)
        acc = accessibility(eblocks, peak_chain, macro_trunc, rng_access)
        null_heights.append(peak_o)
        null_accessibilities.append(acc)
        print(f"  peak {k}: climbed_height={peak_o:.3f} accessibility={acc:.3f}")
    null_heights = np.array(null_heights)
    null_accessibilities = np.array(null_accessibilities)
    print(f"  Null mean climbed_height={null_heights.mean():.3f} std={null_heights.std():.3f}")
    print(f"  Null mean accessibility={null_accessibilities.mean():.3f} std={null_accessibilities.std():.3f}")

    obj_results = {"null_heights": null_heights, "null_access": null_accessibilities, "by_d": {}}

    for d in D_GRID:
        heights, accessibilities, actual_ds = [], [], []
        for k in range(1, N_PER_D + 1):
            rng_gen = np.random.RandomState(d * 1000 + k)
            chain, actual_d = perturb_to_exact_distance(identity_chain, d, rng_gen, n_blocks)
            actual_ds.append(actual_d)
            rng_climb = np.random.RandomState(d * 1000 + k + 5000)
            climbed_chain, climbed_o = hill_climb_on_chain(blocks, chain, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
            rng_access = np.random.RandomState(d * 1000 + k + 10000)
            acc = accessibility(blocks, climbed_chain, macro_trunc, rng_access)
            heights.append(climbed_o)
            accessibilities.append(acc)

        heights = np.array(heights)
        accessibilities = np.array(accessibilities)
        delta_h = heights.mean() - null_heights.mean()
        delta_a = accessibilities.mean() - null_accessibilities.mean()
        pct_h = np.mean([np.mean(null_heights < h) for h in heights])
        pct_a = np.mean([np.mean(null_accessibilities < h) for h in accessibilities])
        obj_results["by_d"][d] = dict(heights=heights, accessibilities=accessibilities,
                                       actual_ds=actual_ds, delta_h=delta_h, delta_a=delta_a,
                                       pct_h=pct_h, pct_a=pct_a)
        print(f"  d={d} (actual d range [{min(actual_ds)},{max(actual_ds)}]): "
              f"height mean={heights.mean():.3f} DeltaH={delta_h:+.3f} A_pct_H={pct_h:.3f}  |  "
              f"access mean={accessibilities.mean():.3f} DeltaA={delta_a:+.3f} A_pct_A={pct_a:.3f}")

    results[name] = obj_results

print("\n" + "=" * 70)
print("CROSS-OBJECT SUMMARY")
print("=" * 70)
for metric_name, key, dkey, pkey in [("raw_climbed_height", "heights", "delta_h", "pct_h"),
                                       ("accessibility", "accessibilities", "delta_a", "pct_a")]:
    print(f"\n--- {metric_name} ---")
    print(f"{'d':>6}" + "".join(f"{name:>12}" for name in objects))
    for d in D_GRID:
        row = [results[name]["by_d"][d][dkey] for name in objects]
        print(f"{d:>6}" + "".join(f"{v:>+12.3f}" for v in row))

np.savez("cache/phase1_results.npz", **{
    f"{name}_{d}_{k}": v for name in objects for d in D_GRID
    for k, v in results[name]["by_d"][d].items() if isinstance(v, np.ndarray)
})
