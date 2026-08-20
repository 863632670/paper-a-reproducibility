"""
G2-2: Paper 101D reproduction. Exact feature set per Paper 101D's own
Method section (G2-2.1, confirmed by direct full-text read):
  [log(tau), H_s, O100, O1000, H_cond, log(mean run length)]
-- explicitly excluding O500 and R2/TE2:TE1 to avoid circularity.
6 features, standardized, PCA + hierarchical (Ward) clustering at
k=2, 3, 4 -- matching Paper 101D's own specified method exactly.

Paper 101D's own reported result (for comparison):
  k=2: 180013 in n=10 cluster (mean O500=-0.041), WITH 180009
  k=3: 180013 in n=9 cluster (mean O500=0.184); 180009 SPLITS OFF alone
  k=4: 180013 in n=6 cluster (mean O500=0.121); 180009 still alone
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import common106 as c
from metrics_tau_hs import tau_int, spectral_entropy, transition_entropy_and_runlength

seeds = list(range(180000, 180024))

print("computing 6-feature set [log(tau), H_s, O100, O1000, H_cond, log(mean_run)]...")
rows = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    t = tau_int(macro)
    h = spectral_entropy(macro)
    o100 = c.O_of_B(source, macro, 100, rng_seed=0)
    o1000 = c.O_of_B(source, macro, 1000, rng_seed=0)
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)  # not a feature, kept for reporting only
    h_cond, mean_run = transition_entropy_and_runlength(macro)
    rows.append((seed, t, h, o100, o1000, h_cond, mean_run, o500))
    print(f"  seed {seed}: logtau={np.log(t):.3f} Hs={h:.3f} O100={o100:.3f} O1000={o1000:.3f} "
          f"Hcond={h_cond:.4f} log(meanrun)={np.log(mean_run):.3f}  [O500={o500:.3f}, not a feature]")

seeds_arr = np.array([r[0] for r in rows])
logtau = np.log(np.array([r[1] for r in rows]))
hs_arr = np.array([r[2] for r in rows])
o100_arr = np.array([r[3] for r in rows])
o1000_arr = np.array([r[4] for r in rows])
hcond_arr = np.array([r[5] for r in rows])
logmeanrun = np.log(np.array([r[6] for r in rows]))
o500_arr = np.array([r[7] for r in rows])

X = np.column_stack([logtau, hs_arr, o100_arr, o1000_arr, hcond_arr, logmeanrun])
X_std = StandardScaler().fit_transform(X)

print("\n=== PCA ===")
pca = PCA(n_components=2)
scores = pca.fit_transform(X_std)
print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)
idx13 = seeds.index(180013)
idx09 = seeds.index(180009)
print(f"180013: PC1={scores[idx13,0]:.3f} PC2={scores[idx13,1]:.3f}   (paper: PC1=3.47, range [-2.49,4.16])")
print(f"180009: PC1={scores[idx09,0]:.3f} PC2={scores[idx09,1]:.3f}")

print("\n=== Hierarchical clustering at k=2,3,4 ===")
for k in [2, 3, 4]:
    hier = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_std)
    cl13 = hier.labels_[idx13]
    cl09 = hier.labels_[idx09]
    n13 = (hier.labels_ == cl13).sum()
    n09 = (hier.labels_ == cl09).sum()
    mean_o500_13 = o500_arr[hier.labels_ == cl13].mean()
    mean_o500_09 = o500_arr[hier.labels_ == cl09].mean()
    print(f"k={k}: 180013 in cluster of size {n13} (mean O500={mean_o500_13:.3f})   "
          f"180009 in cluster of size {n09} (mean O500={mean_o500_09:.3f})   "
          f"180009 is singleton? {n09==1}   same cluster as 180013? {cl13==cl09}")

np.savez("cache/G2_2_results.npz", seeds=seeds_arr, X=X, o500=o500_arr, pc_scores=scores)
