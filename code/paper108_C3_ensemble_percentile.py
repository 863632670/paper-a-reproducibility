"""
C3: S19 ensemble percentile. Question: is S19's phenotype an extreme/
rare configuration relative to a random-peak ensemble, rather than
"complex internal grammar"? Uses the frozen protocol's own
A_percentile(d) metric (Section 4): rank of the historical/perturbed-
at-d accessibility within the N=6 random-peak ensemble's own
accessibility distribution, expressed as a fraction.

No new design -- Phase 1's own per-d heights/accessibilities arrays
(cache/phase1_results.npz) are reused directly. Only the N=6 Null-random
ensemble is regenerated (it was computed but never saved to disk in
Phase 1's own run -- deterministic RNG seeds, identical formula,
regenerating reproduces it exactly, not a new computation).
"""
import numpy as np
import common106 as c
from paper106ab3_5_block_eulerian import block_eulerian_chain

W = 100
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


objects = {"S13": 180013, "S18": 180018, "S19": 180019}
trajs = {}
d13 = np.load("cache/s13_m13.npz")
trajs[180013] = (d13["source13"], d13["macro13"])
for name, seed in objects.items():
    if seed != 180013:
        d = np.load(f"cache/s{seed - 180000:02d}_traj.npz")
        trajs[seed] = (d["source"], d["macro"])

phase1_cache = np.load("cache/phase1_results.npz")

print("=== Regenerating N=6 null-random ensemble (deterministic, matches Phase 1 exactly) ===")
null_data = {}
for name, seed in objects.items():
    source, macro = trajs[seed]
    n_blocks = len(source) // W
    macro_trunc = macro[:n_blocks * W]

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
    null_data[name] = dict(heights=np.array(null_heights), access=np.array(null_accessibilities))
    print(f"{name}: null_heights={null_data[name]['heights']}")
    print(f"{name}: null_access ={null_data[name]['access']}")

print("\n=== A_percentile(d): rank within N=6 null ensemble ===")
results = {}
for name in objects:
    print(f"\n{name}:")
    obj_res = {}
    for d in D_GRID:
        heights = phase1_cache[f"{name}_{d}_heights"]
        accessibilities = phase1_cache[f"{name}_{d}_accessibilities"]
        null_h = null_data[name]["heights"]
        null_a = null_data[name]["access"]
        pct_h = np.mean([np.mean(null_h < h) for h in heights])
        pct_a = np.mean([np.mean(null_a < h) for h in accessibilities])
        obj_res[d] = dict(pct_h=pct_h, pct_a=pct_a, height_mean=heights.mean(), access_mean=accessibilities.mean())
        print(f"  d={d:4d}: A_pct_height={pct_h:.3f}  A_pct_accessibility={pct_a:.3f}  "
              f"(height={heights.mean():.3f}, access={accessibilities.mean():.3f})")
    results[name] = obj_res

print("\n=== Cross-object summary: A_pct_accessibility(d) ===")
print(f"{'d':>6}" + "".join(f"{name:>12}" for name in objects))
for d in D_GRID:
    row = [results[name][d]["pct_a"] for name in objects]
    print(f"{d:>6}" + "".join(f"{v:>12.3f}" for v in row))

print("\n=== S19-specific: is it a rare/extreme configuration? ===")
s19_pcts = [results["S19"][d]["pct_a"] for d in D_GRID]
print(f"S19 A_pct_accessibility across all d: {s19_pcts}")
print(f"  min={min(s19_pcts):.3f} max={max(s19_pcts):.3f} mean={np.mean(s19_pcts):.3f}")
print("  (near 0.5 = typical/unremarkable relative to ensemble; near 0 or 1 = extreme/rare)")
