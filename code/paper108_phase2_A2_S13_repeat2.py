"""
A2-S13 repeat: independent second repeat of the Mode-3 mixed-perturbation
additivity test, S13 ONLY, both distances (d=50, d=188). Same discipline
as A1 -- do not pre-interpret, do not touch Master Map, only decide
whether the candidate interaction (gap << 0, growing with distance)
upgrades to confirmed or regresses to ~0.

Identical pipeline to paper108_phase2_A2_mode3_mixed.py (frozen
protocol's Mode 3 construction, same accessibility pipeline), only
change: new independent SEED_OFFSET (70000, vs. the original run's
50000) and scope narrowed to S13 only -- mirrors exactly how A1 went
from n=1 to n=2 to n=10 by varying only the seed offset.

Original run (offset=50000):
  d=50:  Delta_mixed=+0.520  predicted=+0.672  gap=-0.152
  d=188: Delta_mixed=+0.106  predicted=+0.589  gap=-0.483
"""
import numpy as np
import common106 as c

W = 100
N_PER_D = 15
N_ACCESS_TRIALS = 5
ACCESS_PERTURB = 5
ACCESS_ITER = 80
RANDOM_PEAK_ITER = 120
D_GRID = [50, 188]
BASE_OFFSET = 70000


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


def blocks_from_bin(s_bin, w):
    n_blocks = len(s_bin) // w
    return [s_bin[i * w:(i + 1) * w] for i in range(n_blocks)]


def mode3_mixed_construct(true_bin, w, target_d, n_bit_target, rng, n_blocks):
    identity_chain = list(range(n_blocks))
    chain = list(identity_chain)
    content = true_bin.copy()
    n_bits_flipped = 0
    attempts = 0
    max_attempts = 6000
    while block_aware_distance(chain, identity_chain) < target_d and attempts < max_attempts:
        i, j = rng.choice(n_blocks, size=2, replace=False)
        candidate = list(chain)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cur_d = block_aware_distance(chain, identity_chain)
        cand_d = block_aware_distance(candidate, identity_chain)
        if abs(cand_d - target_d) <= abs(cur_d - target_d):
            chain = candidate
        if n_bits_flipped < n_bit_target:
            pos = rng.choice(len(content))
            content[pos] = 1 - content[pos]
            n_bits_flipped += 1
        attempts += 1
    while n_bits_flipped < n_bit_target:
        pos = rng.choice(len(content))
        content[pos] = 1 - content[pos]
        n_bits_flipped += 1
    blocks_content = blocks_from_bin(content, w)
    final_bin = np.concatenate([blocks_content[i] for i in chain])
    return final_bin, block_aware_distance(chain, identity_chain)


d13 = np.load("cache/s13_m13.npz")
source, macro = d13["source13"], d13["macro13"]
n_blocks = len(source) // W
macro_trunc = macro[:n_blocks * W]
true_bin = (source[:n_blocks * W] > 0).astype(int)
identity_chain = list(range(n_blocks))

d1_cache = np.load("cache/phase2a_results.npz")
d2b_cache = np.load("cache/phase2b_results.npz")
A0 = d2b_cache["S13_0_hist"].mean()

print("=== A2-S13 repeat 2 (offset=70000) ===")
for d in D_GRID:
    mode3_access = []
    for k in range(1, N_PER_D + 1):
        rng_gen = np.random.RandomState(20000 + d * 1000 + k)
        chain1, _ = perturb_to_exact_distance(identity_chain, d, rng_gen, n_blocks)
        blocks_true = blocks_from_bin(true_bin, W)
        source1_bin = np.concatenate([blocks_true[i] for i in chain1])
        n_bit_target = int(np.sum(source1_bin != true_bin))

        rng_mix = np.random.RandomState(BASE_OFFSET + d * 1000 + k)
        final_bin, actual_d = mode3_mixed_construct(true_bin, W, d, n_bit_target, rng_mix, n_blocks)
        blocks3 = blocks_from_bin(final_bin, W)
        chain3 = list(range(len(blocks3)))

        rng_climb = np.random.RandomState(BASE_OFFSET + d * 1000 + k + 5000)
        climbed3, _ = hill_climb_on_chain(blocks3, chain3, macro_trunc, rng_climb, n_iter=RANDOM_PEAK_ITER)
        rng_acc = np.random.RandomState(BASE_OFFSET + d * 1000 + k + 10000)
        acc3 = accessibility(blocks3, climbed3, macro_trunc, rng_acc)
        mode3_access.append(acc3)

    mode3_access = np.array(mode3_access)
    mode1_mean = d1_cache[f"S13_{d}_mode1"].mean()
    mode2_mean = d1_cache[f"S13_{d}_mode2"].mean()
    delta_block = mode1_mean - A0
    delta_bit = mode2_mean - A0
    delta_mixed_actual = mode3_access.mean() - A0
    delta_mixed_predicted = delta_block + delta_bit
    gap = delta_mixed_actual - delta_mixed_predicted

    print(f"\n  d_B={d}:")
    print(f"    Mode3 (mixed) mean: {mode3_access.mean():.3f}")
    print(f"    Delta_block={delta_block:+.3f}  Delta_bit={delta_bit:+.3f}")
    print(f"    Delta_mixed_actual={delta_mixed_actual:+.3f}  predicted={delta_mixed_predicted:+.3f}")
    print(f"    Gap (actual - predicted): {gap:+.3f}")

print("\n=== Comparison across both repeats ===")
print("Repeat 1 (offset=50000): d=50 gap=-0.152, d=188 gap=-0.483")
print("Repeat 2 (offset=70000): see above")
