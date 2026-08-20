"""
S13-G1: Generalization Audit (pre-registered).

Question: does the S13-discovered grammar signature (low H1, real
delta-H second-order structure, source x block-type coupling)
correspond to a broader, predictive object CLASS, or is it S13-specific?

n=3 (S13/S18/S19) is far too small for a genuine leave-one-out
regression -- so this test EXTENDS C2-A/B/C's exact pipeline (same
block fingerprint, same 5-type pooled k-means, same H1/H2/deltaH,
same source-entropy/source-transition/source-block-coupling features)
from 3 objects to the full n=24 confirmed population (180000-180023),
where O500 and PC1 are already known exactly (R100-A, G2-3). This is
the only way to ask the generalization question with real statistical
power, reusing existing infrastructure, not inventing a new protocol.

Features (5, all reused verbatim from C2-A/B/C):
  H1, deltaH (block-type grammar, pooled 5-type k-means across all 24)
  H_source, deltaH_source (source organization)
  block_source_F (source x block-type coupling ANOVA F-stat)

Targets: O500, PC1 (both already exactly known for all 24 trajectories).

Test: simple correlations (n=24, real power) + multi-feature
leave-one-out linear regression predicting O500 and PC1, mirroring
Paper 105A's own LOO test format directly for comparability.

Pre-registered interpretation:
  Case A: S13 sits at a feature extreme, but features don't predict
          O500/PC1 population-wide -> S13 = exceptional case study
  Case B: features show real LOO predictive power -> grammar class
          discovered
  Case C: features show real variance/structure but LOO prediction
          fails -> representation discovery without mechanism
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import Counter
import common106 as c

W = 500
N_CLUSTERS = 5
seeds = list(range(180000, 180024))


def block_fingerprint(macro, m_bin_global, w):
    n_blocks = len(macro) // w
    feats = []
    for i in range(n_blocks):
        seg = macro[i * w:(i + 1) * w]
        seg_bin = m_bin_global[i * w:(i + 1) * w]
        hist = np.bincount(seg_bin, minlength=5) / len(seg_bin)
        mean = seg.mean()
        std = seg.std()
        joint = Counter(zip(seg_bin[:-1].tolist(), seg_bin[1:].tolist()))
        total = len(seg_bin) - 1
        H_joint = -sum((v / total) * np.log2(v / total) for v in joint.values())
        hist_mt = Counter(seg_bin[:-1].tolist())
        H_mt = -sum((v / total) * np.log2(v / total) for v in hist_mt.values())
        trans_entropy = H_joint - H_mt
        feats.append(np.concatenate([hist, [mean, std, trans_entropy]]))
    return np.array(feats)


def cond_entropy(seq, order):
    n = len(seq)
    starts = list(range(order, n - 1))
    if len(starts) < 5:
        return np.nan
    joint_counts = Counter()
    hist_counts = Counter()
    for t in starts:
        hist = tuple(seq[t - order + 1:t + 1])
        nxt = seq[t + 1]
        joint_counts[(hist, nxt)] += 1
        hist_counts[hist] += 1
    total = len(starts)
    H_joint = -sum((v / total) * np.log2(v / total) for v in joint_counts.values())
    H_hist = -sum((v / total) * np.log2(v / total) for v in hist_counts.values())
    return H_joint - H_hist


def source_entropy(source_seg):
    p_pos = np.mean(source_seg > 0)
    p_neg = 1 - p_pos
    ps = [p for p in (p_pos, p_neg) if p > 0]
    return -sum(p * np.log2(p) for p in ps)


print("=== Loading 24-object population, computing block fingerprints ===")
fingerprints = {}
n_blocks_per_obj = {}
sources = {}
macros = {}
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    sources[seed] = source
    macros[seed] = macro
    m_bin_global, _ = c.discretize(macro, n_bins=5)
    fp = block_fingerprint(macro, m_bin_global, W)
    fingerprints[seed] = fp
    n_blocks_per_obj[seed] = len(fp)

print("=== Pooled standardization + 5-type k-means (all 24 objects) ===")
all_feats = np.concatenate([fingerprints[s] for s in seeds], axis=0)
mu, sigma = all_feats.mean(axis=0), all_feats.std(axis=0)
sigma[sigma == 0] = 1.0
all_feats_z = (all_feats - mu) / sigma
km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
all_labels = km.fit_predict(all_feats_z)

labels_per_obj = {}
idx = 0
for seed in seeds:
    n = n_blocks_per_obj[seed]
    labels_per_obj[seed] = all_labels[idx:idx + n]
    idx += n

print("=== Computing H1/deltaH, source features per object ===")
rows = []
for seed in seeds:
    block_labels = labels_per_obj[seed]
    H1 = cond_entropy(block_labels, 1)
    H2 = cond_entropy(block_labels, 2)
    deltaH = H1 - H2

    source = sources[seed]
    n_blocks = n_blocks_per_obj[seed]
    source_trunc = source[:n_blocks * W]
    s_sign = (source_trunc > 0).astype(int)
    H1_src = cond_entropy(s_sign, 1)
    H2_src = cond_entropy(s_sign, 2)
    deltaH_src = H1_src - H2_src

    block_H_source = []
    for i in range(n_blocks):
        seg = source_trunc[i * W:(i + 1) * W]
        block_H_source.append(source_entropy(seg))
    block_H_source = np.array(block_H_source)
    groups_h = [block_H_source[block_labels == t] for t in range(N_CLUSTERS) if np.sum(block_labels == t) >= 2]
    if len(groups_h) >= 2:
        f_stat, _ = stats.f_oneway(*groups_h)
    else:
        f_stat = np.nan

    rows.append((seed, H1, deltaH, deltaH_src, f_stat))
    print(f"seed={seed}: H1={H1:.3f} deltaH={deltaH:+.4f} deltaH_src={deltaH_src:+.4f} block_src_F={f_stat:.2f}")

seeds_arr = np.array([r[0] for r in rows])
H1_arr = np.array([r[1] for r in rows])
deltaH_arr = np.array([r[2] for r in rows])
deltaH_src_arr = np.array([r[3] for r in rows])
F_arr = np.array([r[4] if not np.isnan(r[4]) else 0.0 for r in rows])

o500_data = np.load("cache/R100A_paper94_results.npz")
pc1_data = np.load("cache/G2_3_results.npz")
o500_seeds = o500_data["seeds"]
o500_vals = o500_data["O500"]
pc1_seeds = pc1_data["seeds"]
pc1_vals = pc1_data["pc1"]

o500_map = dict(zip(o500_seeds.tolist(), o500_vals.tolist()))
pc1_map = dict(zip(pc1_seeds.tolist(), pc1_vals.tolist()))
o500_arr = np.array([o500_map[s] for s in seeds_arr])
pc1_arr = np.array([pc1_map[s] for s in seeds_arr])

print("\n=== Simple correlations (n=24) ===")
X = np.column_stack([H1_arr, deltaH_arr, deltaH_src_arr, F_arr])
feat_names = ["H1", "deltaH", "deltaH_src", "block_src_F"]
for name, col in zip(feat_names, X.T):
    r_o, p_o = stats.pearsonr(col, o500_arr)
    r_p, p_p = stats.pearsonr(col, pc1_arr)
    print(f"{name:>12} vs O500: r={r_o:+.3f} p={p_o:.3f}   vs PC1: r={r_p:+.3f} p={p_p:.3f}")

print("\n=== Leave-one-out linear regression (4 features -> O500 / PC1) ===")


def loo_predict(X, y):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        train = [j for j in range(n) if j != i]
        model = LinearRegression()
        model.fit(X[train], y[train])
        preds[i] = model.predict(X[i:i + 1])[0]
    r, p = stats.pearsonr(preds, y)
    return r, p, preds


r_o500, p_o500, preds_o500 = loo_predict(X, o500_arr)
r_pc1, p_pc1, preds_pc1 = loo_predict(X, pc1_arr)
print(f"LOO predict O500: r={r_o500:.3f} p={p_o500:.3f}   (105A's own PC1->O500 LOO: r=0.398 p=0.054, for scale)")
print(f"LOO predict PC1:  r={r_pc1:.3f} p={p_pc1:.3f}")

print("\n=== Is S13 (180013) an outlier on this feature set? ===")
idx13 = seeds_arr.tolist().index(180013)
for name, col in zip(feat_names, X.T):
    pct = 100 * np.mean(col <= col[idx13])
    print(f"{name:>12}: S13 value={col[idx13]:+.4f}  percentile within population={pct:.0f}%")

np.savez("cache/S13G1_results.npz", seeds=seeds_arr, H1=H1_arr, deltaH=deltaH_arr,
         deltaH_src=deltaH_src_arr, block_src_F=F_arr, o500=o500_arr, pc1=pc1_arr,
         preds_o500=preds_o500, preds_pc1=preds_pc1)
