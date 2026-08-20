"""
G2-1A: Paper 97 reproduction, O(B)-only (no tau/H_s -- those come in
G2-1B once implemented against G2-0.5's frozen definitions). Uses the
CORRECTED 24-trajectory population (seeds 180000-180023, exact),
superseding any prior independently-drawn population.

Paper 97's own reported values (n=24, T=2.25):
  O100 vs O500: r=0.588, p=0.0025
  O1000 vs O500: r=0.538, p=0.0067
  O100 vs O1000: r=0.665, p=0.0004
  Hierarchical clustering (2 clusters): cluster means -0.041 / 0.519
  K-means clustering (2 clusters): cluster means 0.537 / -0.217
  Golden pair (180013 +1.99, 180009 -2.06): grouped TOGETHER in both
  clustering solutions -- the specific "failure" this whole G2 pass is
  checking for.
"""
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
import common106 as c

seeds = list(range(180000, 180024))

print("computing O100/O500/O1000 for all 24 corrected trajectories...")
rows = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    o100 = c.O_of_B(source, macro, 100, rng_seed=0)
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)
    o1000 = c.O_of_B(source, macro, 1000, rng_seed=0)
    rows.append((seed, o100, o500, o1000))
    print(f"  seed {seed}: O100={o100:.3f} O500={o500:.3f} O1000={o1000:.3f}")

seeds_arr = np.array([r[0] for r in rows])
o100_arr = np.array([r[1] for r in rows])
o500_arr = np.array([r[2] for r in rows])
o1000_arr = np.array([r[3] for r in rows])

print("\n=== Correlations (n=24, corrected population) ===")
r1, p1 = stats.pearsonr(o100_arr, o500_arr)
print(f"O100 vs O500:  r={r1:.3f} p={p1:.4f}   (paper: r=0.588 p=0.0025)")
r2, p2 = stats.pearsonr(o1000_arr, o500_arr)
print(f"O1000 vs O500: r={r2:.3f} p={p2:.4f}   (paper: r=0.538 p=0.0067)")
r3, p3 = stats.pearsonr(o100_arr, o1000_arr)
print(f"O100 vs O1000: r={r3:.3f} p={p3:.4f}   (paper: r=0.665 p=0.0004)")

print("\n=== Clustering (features: O100, O1000 -- O500 excluded as target, per paper's own design) ===")
X = np.column_stack([o100_arr, o1000_arr])
X_std = StandardScaler().fit_transform(X)

hier = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X_std)
for cl in [0, 1]:
    mask = hier.labels_ == cl
    print(f"Hierarchical cluster {cl}: n={mask.sum()} mean O500={o500_arr[mask].mean():.3f} range=[{o500_arr[mask].min():.3f},{o500_arr[mask].max():.3f}]")

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X_std)
for cl in [0, 1]:
    mask = km.labels_ == cl
    print(f"K-means cluster {cl}: n={mask.sum()} mean O500={o500_arr[mask].mean():.3f} range=[{o500_arr[mask].min():.3f},{o500_arr[mask].max():.3f}]")

print("\n=== Golden pair check: are 180013 and 180009 separated? ===")
idx13 = list(seeds_arr).index(180013)
idx09 = list(seeds_arr).index(180009)
print(f"180013: O500={o500_arr[idx13]:.3f}  hier_cluster={hier.labels_[idx13]}  kmeans_cluster={km.labels_[idx13]}")
print(f"180009: O500={o500_arr[idx09]:.3f}  hier_cluster={hier.labels_[idx09]}  kmeans_cluster={km.labels_[idx09]}")
print(f"Separated by hierarchical clustering? {hier.labels_[idx13] != hier.labels_[idx09]}")
print(f"Separated by k-means clustering? {km.labels_[idx13] != km.labels_[idx09]}")

np.savez("cache/G2_1A_results.npz", seeds=seeds_arr, o100=o100_arr, o500=o500_arr, o1000=o1000_arr,
         hier_labels=hier.labels_, kmeans_labels=km.labels_)
