"""
R100-B: Paper 103B reproduction ("Path Geometry, Not Transition
Topology: The Markov Surrogate Test"). Generates surrogate M-paths from
each trajectory's EXACT empirical first-order transition matrix
(preserving pairwise transition statistics, randomizing path
realization), then computes O500 on original-source + surrogate-macro.

IMPORTANT CAVEAT, flagged before running: this is inherently a
single-realization stochastic test in the original paper (its own
Limitations section says so explicitly: "multiple independent
surrogates... would help characterize surrogate-to-surrogate variance").
An exact numeric match to the paper's specific single draw is not a
coherent target -- this reproduction instead runs N=30 independent
surrogate replicates per trajectory and reports the distribution,
which is a STRONGER test than the paper's own single-draw approach:
checking whether the original O500 is a genuine outlier relative to
the surrogate distribution, not just different from one random draw.

Paper 103B's own single-draw reported values:
  High-O500: 180013 1.993->-0.259, 180019 1.592->0.463, 180018 1.581->-0.111
  Low-O500:  180009 -2.065->-1.962 (stable), 180002 -1.104->-1.490 (same dir),
             180014 -0.511->1.340 (dramatic reversal, unexplained)
"""
import numpy as np
import common106 as c

high_seeds = {180013: 1.993, 180019: 1.592, 180018: 1.581}
low_seeds = {180009: -2.065, 180002: -1.104, 180014: -0.511}
N_REPLICATES = 30


def transition_matrix(macro, n_bins=5):
    m_bin, edges = c.discretize(macro, n_bins=n_bins)
    counts = np.zeros((n_bins, n_bins))
    for a, b in zip(m_bin[:-1], m_bin[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return P, m_bin, edges


def markov_surrogate_macro(macro, n_bins=5, rng=None):
    P, m_bin, edges = transition_matrix(macro, n_bins)
    n = len(m_bin)
    bin_mids = (edges[:-1] + edges[1:]) / 2
    surrogate_bin = np.zeros(n, dtype=int)
    surrogate_bin[0] = m_bin[0]
    for t in range(1, n):
        row = P[surrogate_bin[t - 1]]
        if row.sum() == 0:
            surrogate_bin[t] = surrogate_bin[t - 1]
        else:
            surrogate_bin[t] = rng.choice(n_bins, p=row)
    return bin_mids[surrogate_bin]


all_seeds = {**high_seeds, **low_seeds}
print("Running N=30 Markov-surrogate replicates per trajectory (may take a few minutes)...")
for seed, orig_o500_paper in all_seeds.items():
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    orig_o500 = c.O_of_B(source, macro, 500, rng_seed=0)

    rng = np.random.default_rng(0)
    surrogate_o500s = []
    for rep in range(N_REPLICATES):
        surr_macro = markov_surrogate_macro(macro, rng=rng)
        o = c.O_of_B(source, surr_macro, 500, rng_seed=0)
        surrogate_o500s.append(o)
    surrogate_o500s = np.array(surrogate_o500s)

    group = "HIGH" if seed in high_seeds else "LOW"
    print(f"\n=== seed {seed} ({group}-O500) ===")
    print(f"Original O500 (this reproduction): {orig_o500:.3f}  (paper: {orig_o500_paper:.3f})")
    print(f"Surrogate O500 distribution: mean={surrogate_o500s.mean():.3f} "
          f"std={surrogate_o500s.std():.3f} min={surrogate_o500s.min():.3f} max={surrogate_o500s.max():.3f}")
    single_draw = surrogate_o500s[0]
    print(f"Single-draw surrogate (first replicate): {single_draw:.3f}")
    is_outlier = orig_o500 > surrogate_o500s.max() or orig_o500 < surrogate_o500s.min()
    pctile = 100 * np.mean(surrogate_o500s <= orig_o500)
    print(f"Original O500 percentile within surrogate distribution: {pctile:.1f}%  (outside range: {is_outlier})")
