"""
C2-A execution, per the FROZEN protocol
(C2_A_Transition_Grammar_Mechanism_Protocol.md). Reconstructs
105B-1 (block fingerprinting) -> 105B-2.2 (5-type k-means block
classification) -> 105B-2.2 (H1) -> 105B-2.3 (H2/deltaH), applied
comparatively to S13/S18/S19, to test the mechanistic origin of S19's
"combinatorial" grammar classification.

Frozen choices (see protocol doc for full rationale):
  - block size W=500 (105B's own convention, distinct from Phase 0-2's
    W=100 accessibility work)
  - per-block feature = [5-bin state-distribution histogram (5 feats),
    mean, std, within-block transition entropy] = 8-dim, z-scored
  - discretization: GLOBAL 5-bin edges (common106.discretize), matching
    every other H_cond/run-length computation in this project
  - k-means: 5 clusters, POOLED across all 3 objects' blocks (one
    shared block-type vocabulary), random_state=0, n_init=10
  - H1 = H(B_t+1|B_t), H2 = H(B_t+1|B_t,B_t-1), deltaH = H1-H2
"""
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
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
        # within-block transition entropy H(M_t+1|M_t)
        joint = Counter(zip(seg_bin[:-1].tolist(), seg_bin[1:].tolist()))
        total = len(seg_bin) - 1
        H_joint = -sum((v / total) * np.log2(v / total) for v in joint.values())
        hist_mt = Counter(seg_bin[:-1].tolist())
        H_mt = -sum((v / total) * np.log2(v / total) for v in hist_mt.values())
        trans_entropy = H_joint - H_mt
        feats.append(np.concatenate([hist, [mean, std, trans_entropy]]))
    return np.array(feats)


def cond_entropy(seq, order):
    """H(B_t+1 | B_t, ..., B_t-order+1) via empirical joint counting."""
    n = len(seq)
    starts = range(order, n - 1)
    joint_counts = Counter()
    hist_counts = Counter()
    for t in starts:
        hist = tuple(seq[t - order + 1:t + 1])
        nxt = seq[t + 1]
        joint_counts[(hist, nxt)] += 1
        hist_counts[hist] += 1
    total = len(list(starts))
    H_joint = -sum((v / total) * np.log2(v / total) for v in joint_counts.values())
    H_hist = -sum((v / total) * np.log2(v / total) for v in hist_counts.values())
    return H_joint - H_hist


print("=== Step 1: block fingerprinting (W=500) ===")
fingerprints = {}
n_blocks_per_obj = {}
for name, seed in objects.items():
    source, macro = trajs[seed]
    m_bin_global, _ = c.discretize(macro, n_bins=5)
    fp = block_fingerprint(macro, m_bin_global, W)
    fingerprints[name] = fp
    n_blocks_per_obj[name] = len(fp)
    print(f"{name}: {len(fp)} blocks, feature shape {fp.shape}")

print("\n=== Step 2: pooled standardization + 5-type k-means ===")
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
    counts = np.bincount(labels_per_obj[name], minlength=N_CLUSTERS)
    print(f"{name} block-type distribution: {counts.tolist()}")

print("\n=== Step 3: H1, H2, deltaH per object ===")
results = {}
for name in objects:
    seq = labels_per_obj[name]
    H1 = cond_entropy(seq, order=1)
    H2 = cond_entropy(seq, order=2)
    deltaH = H1 - H2
    results[name] = dict(H1=H1, H2=H2, deltaH=deltaH, n_blocks=len(seq))
    print(f"{name}: H1={H1:.4f}  H2={H2:.4f}  deltaH={deltaH:+.4f}  (n_blocks={len(seq)})")

print("\n=== Cross-object comparison ===")
print(f"{'Object':>8} {'H1':>10} {'H2':>10} {'deltaH':>10}")
for name in objects:
    r = results[name]
    print(f"{name:>8} {r['H1']:>10.4f} {r['H2']:>10.4f} {r['deltaH']:>+10.4f}")

print("\n=== Prediction check ===")
print("Predicted: S13 low H1 + low deltaH ('self-loop')")
print("           S19 high H1 + high deltaH ('combinatorial-with-hidden-grammar')")
print("           S18 either intermediate, or high H1 + low deltaH (no hidden grammar)")

np.savez("cache/C2A_results.npz", **{
    f"{name}_{k}": v for name in objects for k, v in results[name].items()
})
