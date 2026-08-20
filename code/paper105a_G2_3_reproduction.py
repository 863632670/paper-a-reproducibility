"""
G2-3: Paper 105A reproduction on the corrected 24-trajectory
population. Paper 105A's own Method: O(B) at B={20,100,500,1000,5000},
PCA on the 4 NON-CIRCULAR features {20,100,1000,5000} (B=500 excluded
since it IS O500 by definition), plus leave-one-trajectory-out
prediction test.

Paper 105A's own reported values:
  PC1 (59.8% of variance) vs O500: r=0.540, p=0.0065
  LOO linear regression (4-feature): r=0.398, p=0.054 (marginal)
"""
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import common106 as c

seeds = list(range(180000, 180024))
SCALES = [20, 100, 1000, 5000]  # non-circular (excludes B=500)

print("computing O(B) at B=20,100,500(target),1000,5000 for all 24 corrected trajectories...")
rows = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    ob = {B: c.O_of_B(source, macro, B, rng_seed=0) for B in SCALES}
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)
    rows.append((seed, ob[20], ob[100], ob[1000], ob[5000], o500))
    print(f"  seed {seed}: O20={ob[20]:.3f} O100={ob[100]:.3f} O1000={ob[1000]:.3f} O5000={ob[5000]:.3f}  O500={o500:.3f}")

seeds_arr = np.array([r[0] for r in rows])
X = np.column_stack([[r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]])
o500_arr = np.array([r[5] for r in rows])

X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=4)
scores = pca.fit_transform(X_std)
pc1 = scores[:, 0]

print(f"\n=== PCA ===")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"PC1 loadings [O20,O100,O1000,O5000]: {pca.components_[0]}")
r_pc1, p_pc1 = stats.pearsonr(pc1, o500_arr)
print(f"corr(PC1, O500) = r={r_pc1:.3f} p={p_pc1:.4f}   (paper: r=0.540, p=0.0065)")

print(f"\n=== Leave-one-trajectory-out linear regression (4-feature -> O500) ===")
preds = np.zeros(24)
for i in range(24):
    mask = np.ones(24, dtype=bool)
    mask[i] = False
    reg = LinearRegression().fit(X_std[mask], o500_arr[mask])
    preds[i] = reg.predict(X_std[i:i+1])[0]
r_loo, p_loo = stats.pearsonr(preds, o500_arr)
print(f"LOO prediction vs actual O500: r={r_loo:.3f} p={p_loo:.4f}   (paper: r=0.398, p=0.054, marginal)")

print(f"\n=== Golden pair in PC space ===")
idx13 = seeds.index(180013)
idx09 = seeds.index(180009)
print(f"180013: PC1={pc1[idx13]:.3f}  O500={o500_arr[idx13]:.3f}  (rank by PC1: {(-pc1).argsort().tolist().index(idx13)+1}/24)")
print(f"180009: PC1={pc1[idx09]:.3f}  O500={o500_arr[idx09]:.3f}  (rank by PC1: {(-pc1).argsort().tolist().index(idx09)+1}/24)")
d_pc = abs(pc1[idx13] - pc1[idx09])
pc_range = pc1.max() - pc1.min()
print(f"PC1 distance(180013,180009) = {d_pc:.3f}  (full PC1 range = {pc_range:.3f}, {100*d_pc/pc_range:.1f}% of range)")

np.savez("cache/G2_3_results.npz", seeds=seeds_arr, X=X, o500=o500_arr, pc1=pc1, loo_preds=preds)
