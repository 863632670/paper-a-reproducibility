"""
A2: Mode 3 (mixed block+bit perturbation), per the FROZEN protocol
(Paper108_Connectivity_Experiment_Protocol_Frozen.md Section 2):
"interleave Mode 1 and Mode 2 operations (alternate one block-swap, one
bit-flip) until the target d_B is reached, then continue with only
bit-flips to reach a comparable d_H."

No redesign -- same objects (S13, S18, S19), same distance subset
(d_B in {50, 188}, matching Phase 2A exactly), same N=15, same
downstream accessibility pipeline (120-iter hill-climb, then 5-trial
accessibility) as paper108_phase2a_block_breaking.py. Mode 3's
per-draw bit-flip target (n_bit_target) is paired to the SAME target
used for that draw's own Mode-2 counterpart in Phase 2A (recomputed
deterministically from the identical Mode-1 seed formula), extending
Phase 2A's own established pairing convention rather than inventing a
new one.

Core question (additivity test): is Delta(Mode1-Mode3) approximately
Delta(Mode1-Mode2) [i.e. Mode3 behaves like Mode2 alone -- bit-level
perturbation dominates and block-order changes add nothing further],
or does Mode3 diverge from both Mode1 and Mode2 [interaction effect --
combining block-order and bit-level perturbation is not a simple sum
of the two individual effects]?

Additivity is tested relative to each object's own d=0 (true,
unperturbed) accessibility, already measured in Phase 2B:
  Delta_block(d)   = Mode1_access(d) - A0
  Delta_bit(d)     = Mode2_access(d) - A0
  Delta_mixed(d)   = Mode3_access(d) - A0     [measured here]
  Predicted_mixed  = Delta_block(d) + Delta_bit(d)
Close match -> additive/independent dimensions. Large divergence ->
grammar-level interaction.

Topology observation (lightweight, no new infrastructure): the final
climbed chain's block-aware distance from true identity is recorded
for each Mode-3 draw as a first-pass structural indicator (full
self-loop/combinatorial grammar-typing was judged out of scope for
this disciplined, no-redesign pass).
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
BASE_OFFSET = 50000


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


objects = {"S13": 180013, "S18": 180018, "S19": 180019}
trajs = {}
d13 = np.load("cache/s13_m13.npz")
trajs[180013] = (d13["source13"], d13["macro13"])
for name, seed in objects.items():
    if seed != 180013:
        d = np.load(f"cache/s{seed - 180000:02d}_traj.npz")
        trajs[seed] = (d["source"], d["macro"])

d1_cache = np.load("cache/phase2a_results.npz")
d2b_cache = np.load("cache/phase2b_results.npz")

results = {}

for name, seed in objects.items():
    source, macro = trajs[seed]
    n_blocks = len(source) // W
    macro_trunc = macro[:n_blocks * W]
    true_bin = (source[:n_blocks * W] > 0).astype(int)
    identity_chain = list(range(n_blocks))

    print(f"\n{'='*70}\n{name} (seed {seed})\n{'='*70}")

    obj_results = {}
    for d in D_GRID:
        mode3_access = []
        final_dists = []
        for k in range(1, N_PER_D + 1):
            # recompute the exact Mode-1 bit-distance target this draw pairs to
            # (deterministic, same seed formula as paper108_phase2a_block_breaking.py)
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
            final_dists.append(block_aware_distance(climbed3, list(range(len(blocks3)))))

        mode3_access = np.array(mode3_access)
        A0 = d2b_cache[f"{name}_0_hist"].mean()
        mode1_mean = d1_cache[f"{name}_{d}_mode1"].mean()
        mode2_mean = d1_cache[f"{name}_{d}_mode2"].mean()
        delta_block = mode1_mean - A0
        delta_bit = mode2_mean - A0
        delta_mixed_actual = mode3_access.mean() - A0
        delta_mixed_predicted = delta_block + delta_bit

        print(f"\n  d_B={d}:")
        print(f"    A0 (d=0 reference):     {A0:.3f}")
        print(f"    Mode1 (block) mean:     {mode1_mean:.3f}   Delta_block={delta_block:+.3f}")
        print(f"    Mode2 (bit) mean:       {mode2_mean:.3f}   Delta_bit={delta_bit:+.3f}")
        print(f"    Mode3 (mixed) mean:     {mode3_access.mean():.3f}   Delta_mixed_actual={delta_mixed_actual:+.3f}")
        print(f"    Predicted (additive):   Delta_block + Delta_bit = {delta_mixed_predicted:+.3f}")
        print(f"    Additivity gap (actual - predicted): {delta_mixed_actual - delta_mixed_predicted:+.3f}")
        print(f"    Mean final block-distance from identity (Mode3, post-climb): {np.mean(final_dists):.1f}")

        obj_results[d] = dict(mode3=mode3_access, A0=A0, mode1_mean=mode1_mean, mode2_mean=mode2_mean,
                               delta_block=delta_block, delta_bit=delta_bit,
                               delta_mixed_actual=delta_mixed_actual, delta_mixed_predicted=delta_mixed_predicted)
    results[name] = obj_results

print("\n" + "=" * 70)
print("CROSS-OBJECT SUMMARY: additivity gap (Delta_mixed_actual - Delta_mixed_predicted)")
print("=" * 70)
print(f"{'d':>6}" + "".join(f"{name:>14}" for name in objects))
for d in D_GRID:
    row = [results[name][d]["delta_mixed_actual"] - results[name][d]["delta_mixed_predicted"] for name in objects]
    print(f"{d:>6}" + "".join(f"{v:>+14.3f}" for v in row))

np.savez("cache/A2_mode3_results.npz", **{
    f"{name}_{d}_{k}": v for name in objects for d in D_GRID
    for k, v in results[name][d].items() if isinstance(v, np.ndarray)
})
