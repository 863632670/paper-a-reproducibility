"""
R100-A: Paper 98 reproduction ("The Fourth Candidate Also Fails: Higher-
Order Memory Does Not Predict O500"). Tests H_k = H(M_{t+1}|M_t,...,
M_{t-k+1}) for k=1,2,5,10 on Paper 94's top-6/bottom-6 O500 extreme
groups, relative reduction (H_1-H_k)/H_1.

Paper 98's own reported values (bottom-6/low-O500 vs top-6/high-O500):
  frac2:  0.085 vs 0.052
  frac5:  0.165 vs 0.124
  frac10: 0.284 vs 0.219   (k=10 subsampled to 30000 points -- paper's
                             own subsample RNG is unknown, so this
                             specific value is Level 2 at best)

Top-6 (O500 high): 180013, 180019, 180018, 180007, 180020, 180022
Bottom-6 (O500 low): 180023, 180005, 180010, 180014, 180002, 180009
"""
import numpy as np
from scipy import stats
from collections import Counter
import common106 as c

top6 = [180013, 180019, 180018, 180007, 180020, 180022]
bot6 = [180023, 180005, 180010, 180014, 180002, 180009]


def cond_entropy_history(m_bin, k, max_points=30000, rng=None):
    n = len(m_bin)
    starts = np.arange(k - 1, n - 1)
    if len(starts) > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        starts = rng.choice(starts, size=max_points, replace=False)
        starts.sort()
    joint_counts = Counter()
    hist_counts = Counter()
    for t in starts:
        hist = tuple(m_bin[t - k + 1:t + 1])
        nxt = m_bin[t + 1]
        joint_counts[(hist, nxt)] += 1
        hist_counts[hist] += 1
    total = len(starts)
    H_joint = -sum((v / total) * np.log2(v / total) for v in joint_counts.values())
    H_hist = -sum((v / total) * np.log2(v / total) for v in hist_counts.values())
    return H_joint - H_hist


def get_macro(seed):
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    m_bin, _ = c.discretize(d["macro"], n_bins=5)
    return m_bin


results = {}
for grp_name, seeds in [("top6", top6), ("bot6", bot6)]:
    fracs = {2: [], 5: [], 10: []}
    for seed in seeds:
        m_bin = get_macro(seed)
        H1 = cond_entropy_history(m_bin, 1)
        for k in (2, 5, 10):
            rng = np.random.default_rng(0)
            Hk = cond_entropy_history(m_bin, k, rng=rng)
            frac = (H1 - Hk) / H1
            fracs[k].append(frac)
    results[grp_name] = fracs
    print(f"{grp_name}: H1 range check done")

print("\n=== Paper 98 core test: mean relative entropy reduction ===")
for k in (2, 5, 10):
    top_mean = np.mean(results["top6"][k])
    bot_mean = np.mean(results["bot6"][k])
    t, p = stats.ttest_ind(results["bot6"][k], results["top6"][k])
    label = {2: "frac2", 5: "frac5", 10: "frac10 (Level-2, subsample RNG unknown)"}[k]
    paper = {2: "0.085 vs 0.052", 5: "0.165 vs 0.124", 10: "0.284 vs 0.219"}[k]
    print(f"{label}: bottom6={bot_mean:.3f} top6={top_mean:.3f}  (paper: {paper})  t-test p={p:.3f}")

print("\n=== Direction check (paper's core claim: bottom6 > top6 at every k) ===")
for k in (2, 5, 10):
    print(f"k={k}: bottom6 > top6 ? {np.mean(results['bot6'][k]) > np.mean(results['top6'][k])}")
