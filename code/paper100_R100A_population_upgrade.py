"""
R100-A: Paper 100 population-claim upgrade. Reruns
`paper100_population_pattern.py`'s exact test battery, but on the now-
CONFIRMED-correct 24-seed population (180000-180023), replacing the
previous independently-drawn gen_pop24.py population (which shared
only the golden pair + a few seeds with the real population, per
Paper100_Population_Pattern_Reconstruction.md's own Level-2 caveat).
This upgrades the population test from Level 2 (directional, different
sample) toward Level 1 (same sample as the original paper), if the
numbers land close to the paper's own r=0.365/p=0.080 and 2.568/1.976
group means.

Paper 100's own reported values:
  TE2/TE1 vs O500: r=0.365, p=0.080
  Top8/Bottom8 group means: 2.568 vs 1.976, p=0.274
"""
import numpy as np
from scipy import stats
import common106 as c
from paper100_goldenpair_te import discretize_pair, TE_k

seeds = list(range(180000, 180024))

results = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)
    s_bin, m_bin = discretize_pair(source, macro)
    te1 = TE_k(s_bin, m_bin, 1)
    te2 = TE_k(s_bin, m_bin, 2)
    ratio = te2 / te1 if te1 else float("nan")
    results.append((seed, o500, te1, te2, ratio))
    print(f"seed={seed:6d}  O500={o500:7.3f}  TE1={te1:.6f}  TE2={te2:.6f}  TE2/TE1={ratio:.3f}")

seeds_arr = np.array([r[0] for r in results])
o500s = np.array([r[1] for r in results])
te1s = np.array([r[2] for r in results])
te2s = np.array([r[3] for r in results])
ratios = np.array([r[4] for r in results])

print("\n=== Correlations (n=24, CORRECTED population 180000-180023) ===")
r_te1, p_te1 = stats.pearsonr(te1s, o500s)
print(f"TE1 vs O500:        r={r_te1:.3f}  p={p_te1:.3f}")
r_ratio, p_ratio = stats.pearsonr(ratios, o500s)
print(f"TE2/TE1 vs O500:    r={r_ratio:.3f}  p={p_ratio:.3f}   (paper: r=0.365, p=0.080)")

print("\n=== Top-8 vs bottom-8 O500 group comparison ===")
order = np.argsort(-o500s)
top8_idx = order[:8]
bot8_idx = order[-8:]
top8_ratio = ratios[top8_idx].mean()
bot8_ratio = ratios[bot8_idx].mean()
t, p = stats.ttest_ind(ratios[top8_idx], ratios[bot8_idx])
print(f"Top8 mean TE2/TE1={top8_ratio:.3f}  Bottom8 mean TE2/TE1={bot8_ratio:.3f}  (paper: 2.568 vs 1.976)")
print(f"t-test: t={t:.3f} p={p:.3f}   (paper: p=0.274)")

print("\n=== Golden pair outlier check ===")
gp_idx13 = seeds.index(180013)
gp_idx09 = seeds.index(180009)
rank_o500_13 = int((-o500s).argsort().tolist().index(gp_idx13)) + 1
rank_o500_09 = int((-o500s).argsort().tolist().index(gp_idx09)) + 1
rank_ratio_13 = int((-ratios).argsort().tolist().index(gp_idx13)) + 1
rank_ratio_09 = int((-ratios).argsort().tolist().index(gp_idx09)) + 1
print(f"180013: O500={o500s[gp_idx13]:.3f} (rank {rank_o500_13}/24), TE2/TE1={ratios[gp_idx13]:.3f} (rank {rank_ratio_13}/24)")
print(f"180009: O500={o500s[gp_idx09]:.3f} (rank {rank_o500_09}/24), TE2/TE1={ratios[gp_idx09]:.3f} (rank {rank_ratio_09}/24)")

np.savez("cache/R100A_paper100_population_results.npz", seeds=seeds_arr, o500=o500s, te1=te1s, te2=te2s, ratio=ratios)
