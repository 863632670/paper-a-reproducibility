"""
C2-C: Source Organization Audit, per your 3-level design. Question:
is source imbalance a DISCRIMINATING representation variable across
S13/S18/S19 (not yet "is it the cause of S19's phenotype" -- exclusion/
discrimination test first, same discipline as C2-B).

Level 1 (static source imbalance): per-block source spin fraction and
  entropy H_source = -[p+log2(p+) + p-log2(p-)].
Level 2 (source transition structure): source's own H1/H2/deltaH,
  computed globally (primary, high sample size) AND per-block
  (secondary, feeds Level 3).
Level 3 (source x block interaction): does per-block source_entropy /
  source_deltaH vary systematically ACROSS the 5 block-type labels
  (reused verbatim from C2-A/B's deterministic k-means fit, not
  re-fit)? Tests discrimination capability, not causal direction.
"""
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy import stats
import common106 as c

W = 500
N_CLUSTERS = 5

objects = {"S13": 180013, "S18": 180018, "S19": 180019}
trajs = {}
d13 = np.load("cache/s13_m13.npz")
trajs[180013] = (d13["source13"], d13["macro13"])
for name, seed in objects.items():
    if seed != 180013:
        d = np.load(f"cache/s{seed - 180000:02d}_traj.npz")
        trajs[seed] = (d["source"], d["macro"])


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
    return -sum(p * np.log2(p) for p in ps), max(p_pos, p_neg)


# --- Regenerate block-type labels, identical to C2-A/B (deterministic) ---
print("=== Regenerating block-type labels (identical to C2-A/B) ===")
fingerprints = {}
n_blocks_per_obj = {}
for name, seed in objects.items():
    source, macro = trajs[seed]
    m_bin_global, _ = c.discretize(macro, n_bins=5)
    fp = block_fingerprint(macro, m_bin_global, W)
    fingerprints[name] = fp
    n_blocks_per_obj[name] = len(fp)

all_feats = np.concatenate([fingerprints[name] for name in objects], axis=0)
mu, sigma = all_feats.mean(axis=0), all_feats.std(axis=0)
sigma[sigma == 0] = 1.0
all_feats_z = (all_feats - mu) / sigma
km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
all_labels = km.fit_predict(all_feats_z)

labels_per_obj = {}
idx = 0
for name in objects:
    n = n_blocks_per_obj[name]
    labels_per_obj[name] = all_labels[idx:idx + n]
    idx += n

# --- Level 1 & 2: per-object static + global source structure ---
print("\n=== Level 1: static source imbalance (global, per object) ===")
level1 = {}
for name, seed in objects.items():
    source, macro = trajs[seed]
    n_blocks = len(source) // W
    source_trunc = source[:n_blocks * W]
    H_src, dominance = source_entropy(source_trunc)
    level1[name] = dict(H_source=H_src, dominance=dominance)
    print(f"{name}: H_source={H_src:.4f} bits  dominance(max p)={dominance:.4f}")

print("\n=== Level 2: source transition structure (GLOBAL, primary) ===")
level2 = {}
for name, seed in objects.items():
    source, macro = trajs[seed]
    n_blocks = len(source) // W
    source_trunc = source[:n_blocks * W]
    s_sign = (source_trunc > 0).astype(int)
    H1 = cond_entropy(s_sign, 1)
    H2 = cond_entropy(s_sign, 2)
    deltaH = H1 - H2
    level2[name] = dict(H1_source=H1, H2_source=H2, deltaH_source=deltaH)
    print(f"{name}: H1_source={H1:.4f}  H2_source={H2:.4f}  deltaH_source={deltaH:+.4f}")

# --- Level 3: source x block-type interaction (per-block features, grouped by block-type) ---
print("\n=== Level 3: source x block-type interaction ===")
level3 = {}
for name, seed in objects.items():
    source, macro = trajs[seed]
    n_blocks = len(source) // W
    source_trunc = source[:n_blocks * W]
    block_H_source = []
    block_dominance = []
    for i in range(n_blocks):
        seg = source_trunc[i * W:(i + 1) * W]
        h, dom = source_entropy(seg)
        block_H_source.append(h)
        block_dominance.append(dom)
    block_H_source = np.array(block_H_source)
    block_dominance = np.array(block_dominance)
    labels = labels_per_obj[name]

    groups_h = [block_H_source[labels == t] for t in range(N_CLUSTERS) if np.sum(labels == t) >= 2]
    if len(groups_h) >= 2:
        f_stat, p_val = stats.f_oneway(*groups_h)
    else:
        f_stat, p_val = np.nan, np.nan

    print(f"\n{name}: per-block H_source by block-type (mean +/- std, n):")
    for t in range(N_CLUSTERS):
        mask = labels == t
        if mask.sum() > 0:
            print(f"  type {t}: H_source={block_H_source[mask].mean():.3f}+/-{block_H_source[mask].std():.3f}  "
                  f"dominance={block_dominance[mask].mean():.3f}  n={mask.sum()}")
    print(f"  One-way ANOVA across block-types: F={f_stat:.3f} p={p_val:.4f}")
    level3[name] = dict(f_stat=f_stat, p_val=p_val)

print("\n=== Cross-object summary ===")
print(f"{'Object':>8} {'H_source':>10} {'dominance':>10} {'H1_src':>9} {'dH_src':>9} {'block-type F':>13} {'p':>8}")
for name in objects:
    l1, l2, l3 = level1[name], level2[name], level3[name]
    print(f"{name:>8} {l1['H_source']:>10.4f} {l1['dominance']:>10.4f} {l2['H1_source']:>9.4f} "
          f"{l2['deltaH_source']:>+9.4f} {l3['f_stat']:>13.3f} {l3['p_val']:>8.4f}")

np.savez("cache/C2C_results.npz",
         **{f"{name}_l1_{k}": v for name in objects for k, v in level1[name].items()},
         **{f"{name}_l2_{k}": v for name in objects for k, v in level2[name].items()},
         **{f"{name}_l3_{k}": v for name in objects for k, v in level3[name].items()})
