"""
108 Connectivity Redesign, Phase 2A: block-breaking vs block-preserving
perturbation, per your Phase 2A design. Tests whether Phase 1's
neighborhood-accessibility structure depends on BLOCK organization
(connecting to 106's core finding) or would appear under any
same-raw-distance perturbation regardless of block structure.

Distances tested: d_B in {50, 188} (skips d=0, where both modes are
trivially identical to the true sequence). N=15 per (object, distance,
mode). Same symmetric protocol as Phase 1 (120-iter hill-climb, then
5-trial accessibility).

Mode 1 (block-preserving): block-pair swaps to reach exact target d_B
(block content/boundaries fully intact, per Phase 1/Amendment 1).
Mode 2 (block-breaking): flip the SAME NUMBER of raw bits (measured from
a paired Mode-1 realization's actual bit-Hamming distance) at uniformly
random individual positions, ignoring block boundaries entirely -- same
nominal raw distance from true, block structure destroyed. The resulting
sequence is then re-partitioned into fresh W=100 blocks and subjected to
the identical downstream hill-climb/accessibility protocol.
"""
import numpy as np
import common106 as c
from paper106ab3_5_block_eulerian import block_eulerian_chain

W = 100
N_PER_D = 15
N_ACCESS_TRIALS = 5
ACCESS_PERTURB = 5
ACCESS_ITER = 80
RANDOM_PEAK_ITER = 120
D_GRID = [50, 188]


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


def bit_flip_perturb(true_source_bin, n_flips, rng):
    """Mode 2: flip n_flips random individual bit positions, ignoring block boundaries."""
    new_bin = true_source_bin.copy()
    positions = rng.choice(len(new_bin), size=n_flips, replace=False)
    new_bin[positions] = 1 - new_bin[positions]
    return new_bin


def blocks_from_bin(s_bin, w):
    n_blocks = len(s_bin) // w
    return [s_bin[i * w:(i + 1) * w] for i in range(n_blocks)]


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
    true_bin = (source[:n_blocks * W] > 0).astype(int)
    blocks_true = blocks_from_bin(true_bin, W)
    identity_chain = list(range(n_blocks))

    print(f"\n{'='*70}\n{name} (seed {seed})\n{'='*70}")

    obj_results = {}
    for d in D_GRID:
        print(f"\n  d_B={d}:")
        mode1_access, mode2_access, bit_distances = [], [], []
        for k in range(1, N_PER_D + 1):
            # Mode 1: block-preserving
            rng_gen = np.random.RandomState(20000 + d * 1000 + k)
            chain1, actual_d = perturb_to_exact_distance(identity_chain, d, rng_gen, n_blocks)
            source1_bin = np.concatenate([blocks_true[i] for i in chain1])
            n_bit_diff = int(np.sum(source1_bin != true_bin))
            bit_distances.append(n_bit_diff)

            rng_climb1 = np.random.RandomState(20000 + d * 1000 + k + 5000)
            climbed1, _ = hill_climb_on_chain(blocks_true, chain1, macro_trunc, rng_climb1, n_iter=RANDOM_PEAK_ITER)
            rng_acc1 = np.random.RandomState(20000 + d * 1000 + k + 10000)
            acc1 = accessibility(blocks_true, climbed1, macro_trunc, rng_acc1)
            mode1_access.append(acc1)

            # Mode 2: block-breaking, same nominal bit-distance as this Mode-1 draw
            rng_bits = np.random.RandomState(20000 + d * 1000 + k + 15000)
            source2_bin = bit_flip_perturb(true_bin, n_bit_diff, rng_bits)
            blocks2 = blocks_from_bin(source2_bin, W)
            chain2 = list(range(len(blocks2)))
            rng_climb2 = np.random.RandomState(20000 + d * 1000 + k + 20000)
            climbed2, _ = hill_climb_on_chain(blocks2, chain2, macro_trunc, rng_climb2, n_iter=RANDOM_PEAK_ITER)
            rng_acc2 = np.random.RandomState(20000 + d * 1000 + k + 25000)
            acc2 = accessibility(blocks2, climbed2, macro_trunc, rng_acc2)
            mode2_access.append(acc2)

        mode1_access = np.array(mode1_access)
        mode2_access = np.array(mode2_access)
        delta_block = mode1_access.mean() - mode2_access.mean()
        print(f"    bit-distances (mode1 realizations): mean={np.mean(bit_distances):.0f} "
              f"range=[{min(bit_distances)},{max(bit_distances)}]")
        print(f"    Mode1 (block-preserving) accessibility: mean={mode1_access.mean():.3f} std={mode1_access.std():.3f}")
        print(f"    Mode2 (block-breaking)   accessibility: mean={mode2_access.mean():.3f} std={mode2_access.std():.3f}")
        print(f"    Delta(Mode1 - Mode2) = {delta_block:+.3f}")
        obj_results[d] = dict(mode1=mode1_access, mode2=mode2_access, delta_block=delta_block)

    results[name] = obj_results

print("\n" + "=" * 70)
print("CROSS-OBJECT SUMMARY: Delta(block-preserved - block-broken)")
print("=" * 70)
print(f"{'d':>6}" + "".join(f"{name:>12}" for name in objects))
for d in D_GRID:
    row = [results[name][d]["delta_block"] for name in objects]
    print(f"{d:>6}" + "".join(f"{v:>+12.3f}" for v in row))

np.savez("cache/phase2a_results.npz", **{
    f"{name}_{d}_{k}": v for name in objects for d in D_GRID
    for k, v in results[name][d].items() if isinstance(v, np.ndarray)
})
