"""
Reconstruction of Paper 105A-Ext's SPECIFIC additional claims beyond what
paper105b0_pc1_transition.py already covers (baseline PC1 across 8 scales,
fine-scale collapse toward B=2000 -- already run, pattern reproduces).

This script adds the two things 105A-Ext claims that 105B-0's own script
doesn't test:
  1. Leave-ONE-SCALE-out robustness: does the PC1-O500 correlation survive
     removing any single one of the 8 non-circular scales?
  2. The B=5000 REVERSAL: does PC1's correlation with the raw O(B) curve
     go from strongly positive (B=10-1000) to null (B=2000) to NEGATIVE
     (B=5000), as 105A-Ext reports (r=0.015 at B=2000, r=-0.495 at B=5000)?

Same population as paper105b0_pc1_transition.py (gen_pop24.py's 24
trajectories, Level-2 ceiling -- Paper 94's original seeds unrecoverable).

Paper 105A-Ext's reported values:
  Leave-one-scale-out: all 8 estimates r=0.510-0.591, all p<0.011
  PC1 vs O(B) at B=10,20,50,100,250,500,1000: r=0.862,0.872,0.898,0.873,0.842,(0.552 vs O500 itself),0.678
  PC1 vs O(B) at B=2000: r=0.015, p=0.946 (vanishes)
  PC1 vs O(B) at B=5000: r=-0.495, p=0.014 (reverses)
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import common106 as c

seeds = np.load("cache/pop24_seeds.npy")
trajs = {}
for seed in seeds:
    suffix = int(seed) - 180000
    if suffix == 13:
        d = np.load("cache/s13_m13.npz")
        trajs[int(seed)] = (d["source13"], d["macro13"])
    else:
        d = np.load(f"cache/s{suffix:02d}_traj.npz")
        trajs[int(seed)] = (d["source"], d["macro"])

scales_8 = [10, 20, 50, 100, 250, 1000, 2000, 5000]

print("computing O(B) at 8 non-circular scales + O500 (B=500) for all 24 trajectories...")
X = np.zeros((len(seeds), len(scales_8)))
o500 = np.zeros(len(seeds))
for i, seed in enumerate(seeds):
    source, macro = trajs[int(seed)]
    for j, B in enumerate(scales_8):
        X[i, j] = c.O_of_B(source, macro, B=B, rng_seed=0)
    o500[i] = c.O_of_B(source, macro, B=500, rng_seed=0)


def pc1_scores_signed(X_sub, ref_col_idx_in_sub):
    """PCA PC1, sign-fixed so loading on the reference column is positive
    (PCA sign is otherwise arbitrary -- must be pinned for comparability
    across leave-one-out folds)."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_sub)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X_std)[:, 0]
    loading = pca.components_[0]
    if loading[ref_col_idx_in_sub] < 0:
        scores = -scores
    return scores


print("\n=== Leave-one-scale-out: PC1-O500 correlation with each of the 8 scales removed ===")
loo_results = []
for drop_idx, drop_B in enumerate(scales_8):
    keep_idx = [i for i in range(8) if i != drop_idx]
    X_sub = X[:, keep_idx]
    # reference column: use whichever kept column is closest to B=100 (a
    # scale far from the transition, expected to load positively throughout)
    ref_candidates = [k for k, i in enumerate(keep_idx) if scales_8[i] == 100]
    ref_col = ref_candidates[0] if ref_candidates else 0
    scores = pc1_scores_signed(X_sub, ref_col)
    r, p = pearsonr(scores, o500)
    loo_results.append((drop_B, r, p))
    print(f"  drop B={drop_B:>5}: PC1-O500 r={r:.3f}  p={p:.3f}   (paper's full range: r=0.510-0.591, all p<0.011)")

print("\n=== Full PC1 (all 8 scales, baseline) vs. O(B) at each scale, including B=5000 reversal check ===")
scores_full = pc1_scores_signed(X, scales_8.index(100))
r500, p500 = pearsonr(scores_full, o500)
print(f"  PC1 vs O500 (B=500, held out of PCA construction): r={r500:.3f} p={p500:.3f}   (paper: r=0.552)")
paper_curve = {10: 0.862, 20: 0.872, 50: None, 100: 0.898, 250: 0.873, 1000: 0.842, 2000: 0.015, 5000: -0.495}
for B in scales_8:
    ob_vals = np.array([c.O_of_B(trajs[int(s)][0], trajs[int(s)][1], B=B, rng_seed=0) for s in seeds])
    r, p = pearsonr(scores_full, ob_vals)
    pv = paper_curve.get(B)
    print(f"  PC1 vs O(B={B:>5}): r={r:.3f} p={p:.3f}   (paper: {pv})")
