"""
R100-A: Paper 95 reproduction ("Sticky Transition Structure: The Golden
Pair's Layer 2 Case Study"). This is a golden-pair (n=2) case study, not
a population correlation -- tests the specific structural transition-
matrix claims for seeds 180013 (high-O500) and 180009 (low-O500).

Paper 95's own reported values:
  180013's most-extreme bin: self-transition probability = 1.000 (perfect)
  180009's second-most-extreme bin: "leak" probability = 0.108, with no
    counterpart anywhere in 180013's matrix
  Overall self-transition persistence: 180013 = 0.9965, 180009 = 0.9927
"""
import numpy as np
import common106 as c

np.set_printoptions(precision=4, suppress=True)


def transition_matrix(macro, n_bins=5):
    m_bin, edges = c.discretize(macro, n_bins=n_bins)
    counts = np.zeros((n_bins, n_bins))
    for a, b in zip(m_bin[:-1], m_bin[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return P, counts


def overall_self_transition_rate(counts):
    return float(np.trace(counts) / counts.sum())


for seed in [180013, 180009]:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    macro = d["macro"]
    P, counts = transition_matrix(macro)
    persistence = overall_self_transition_rate(counts)
    print(f"\n=== seed {seed} ===")
    print("Transition matrix P(bin_t+1 | bin_t):")
    print(P)
    print("Diagonal (self-transition probs per bin):", np.diag(P))
    print(f"Overall self-transition persistence (trace(counts)/total): {persistence:.4f}")

print("\n=== Paper 95 claims check ===")
print("180013 most-extreme bin self-transition -- paper: 1.000")
print("180009 second-most-extreme bin leak -- paper: 0.108, no counterpart in 180013")
print("Overall persistence -- paper: 180013=0.9965, 180009=0.9927")
