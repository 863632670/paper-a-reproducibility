"""
G2-1B.2: full Paper 97 reproduction with all 4 features [O100, O1000,
tau_int, H_s] -- the actual fair test of whether the golden pair
fails to separate under clustering, now that G2-1A confirmed O(B) is
exact and G2-1B.1 sanity-checked tau_int/H_s.

Paper 97's own reported values (for comparison):
  Hierarchical clustering: cluster 0 n=10 mean=-0.041, cluster 1 n=14 mean=0.519
  K-means: cluster 0 n=16 mean=0.537, cluster 1 n=8 mean=-0.217
  Golden pair (180013 +1.99, 180009 -2.06) grouped TOGETHER in both.
"""
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import common106 as c
from metrics_tau_hs import tau_int, spectral_entropy

seeds = list(range(180000, 180024))

print("computing full feature set [O100, O500(target), O1000, tau_int, H_s]...")
rows = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    o100 = c.O_of_B(source, macro, 100, rng_seed=0)
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)
    o1000 = c.O_of_B(source, macro, 1000, rng_seed=0)
    t = tau_int(macro)
    h = spectral_entropy(macro)
    rows.append((seed, o100, o500, o1000, t, h))

seeds_arr = np.array([r[0] for r in rows])
o100_arr = np.array([r[1] for r in rows])
o500_arr = np.array([r[2] for r in rows])
o1000_arr = np.array([r[3] for r in rows])
tau_arr = np.array([r[4] for r in rows])
hs_arr = np.array([r[5] for r in rows])

# Feature matrix: [O100, O1000, tau_int, H_s] -- O500 excluded as target, matching Paper 97's design
X = np.column_stack([o100_arr, o1000_arr, tau_arr, hs_arr])
X_std = StandardScaler().fit_transform(X)

print("\n=== PCA ===")
pca = PCA(n_components=4)
scores = pca.fit_transform(X_std)
pc1 = scores[:, 0]
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("PC1 loadings [O100, O1000, tau_int, H_s]:", pca.components_[0])
r_pc1, p_pc1 = stats.pearsonr(pc1, o500_arr)
print(f"corr(PC1, O500) = r={r_pc1:.3f} p={p_pc1:.4f}")

print("\n=== Golden pair distance in 4D standardized feature space ===")
idx13 = seeds.index(180013)
idx09 = seeds.index(180009)
d_euclid = np.linalg.norm(X_std[idx13] - X_std[idx09])
dists = [np.linalg.norm(X_std[i] - X_std[j]) for i in range(24) for j in range(i + 1, 24)]
print(f"d(180013, 180009) = {d_euclid:.3f}")
print(f"Population pairwise distance distribution: mean={np.mean(dists):.3f} median={np.median(dists):.3f} "
      f"min={np.min(dists):.3f} max={np.max(dists):.3f}")
print(f"Is d(180013,180009) among the largest pairwise distances? "
      f"percentile={100*np.mean(np.array(dists) <= d_euclid):.1f}%")

print("\n=== Clustering (4 features: O100, O1000, tau_int, H_s) ===")
hier = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X_std)
for cl in [0, 1]:
    mask = hier.labels_ == cl
    print(f"Hierarchical cluster {cl}: n={mask.sum()} mean O500={o500_arr[mask].mean():.3f}")

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X_std)
for cl in [0, 1]:
    mask = km.labels_ == cl
    print(f"K-means cluster {cl}: n={mask.sum()} mean O500={o500_arr[mask].mean():.3f}")

print("\n=== Golden pair separation check (the actual R1 question) ===")
print(f"180013: O500={o500_arr[idx13]:.3f}  hier_cluster={hier.labels_[idx13]}  kmeans_cluster={km.labels_[idx13]}")
print(f"180009: O500={o500_arr[idx09]:.3f}  hier_cluster={hier.labels_[idx09]}  kmeans_cluster={km.labels_[idx09]}")
print(f"Separated by hierarchical clustering? {hier.labels_[idx13] != hier.labels_[idx09]}")
print(f"Separated by k-means clustering? {km.labels_[idx13] != km.labels_[idx09]}")

np.savez("cache/G2_1B2_results.npz", seeds=seeds_arr, o100=o100_arr, o500=o500_arr, o1000=o1000_arr,
         tau_int=tau_arr, H_s=hs_arr, pc1=pc1, hier_labels=hier.labels_, kmeans_labels=km.labels_)
