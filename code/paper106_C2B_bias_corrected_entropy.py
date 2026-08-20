"""
C2-B: bias-corrected transition grammar estimate. Same data, same block
fingerprint, same 5-type pooled k-means as C2-A (deterministic,
random_state=0 -> re-running reproduces identical block-type labels,
no redesign). Adds:
  1. Miller-Madow bias correction to H1/H2/deltaH (analytical).
  2. Moving-block bootstrap (block length L=10, N=500 replicates) for
     confidence intervals on deltaH per object -- tests directly
     whether the C2-A ranking (S18 highest, S19 lowest) is stable
     under resampling or an artifact of n_blocks=75's small sample.

Question this answers: is S18's high deltaH real, or sample-size
inflation? Is S19's low deltaH stable, or sample-size compression?
"""
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import common106 as c

W = 500
N_CLUSTERS = 5
BOOT_BLOCK_LEN = 10
N_BOOTSTRAP = 500

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


def entropy_raw(counter, total):
    probs = np.array([v / total for v in counter.values()])
    return -np.sum(probs * np.log2(probs))


def entropy_mm(counter, total):
    H_ml = entropy_raw(counter, total)
    m = len(counter)
    return H_ml + (m - 1) / (2 * total * np.log(2))


def cond_entropy(seq, order, corrected=False):
    n = len(seq)
    starts = list(range(order, n - 1))
    joint_counts = Counter()
    hist_counts = Counter()
    for t in starts:
        hist = tuple(seq[t - order + 1:t + 1])
        nxt = seq[t + 1]
        joint_counts[(hist, nxt)] += 1
        hist_counts[hist] += 1
    total = len(starts)
    if total == 0:
        return np.nan
    fn = entropy_mm if corrected else entropy_raw
    return fn(joint_counts, total) - fn(hist_counts, total)


def moving_block_bootstrap(seq, block_len, n_boot, rng):
    n = len(seq)
    n_possible_starts = n - block_len + 1
    deltas = []
    for _ in range(n_boot):
        resampled = []
        while len(resampled) < n:
            start = rng.randint(0, n_possible_starts)
            resampled.extend(seq[start:start + block_len])
        resampled = np.array(resampled[:n])
        H1 = cond_entropy(resampled, 1)
        H2 = cond_entropy(resampled, 2)
        deltas.append(H1 - H2)
    return np.array(deltas)


print("=== Regenerating block-type labels (identical to C2-A, deterministic) ===")
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

print("\n=== Step 1: Miller-Madow bias-corrected H1/H2/deltaH ===")
mm_results = {}
for name in objects:
    seq = labels_per_obj[name]
    H1_raw = cond_entropy(seq, 1, corrected=False)
    H2_raw = cond_entropy(seq, 2, corrected=False)
    H1_mm = cond_entropy(seq, 1, corrected=True)
    H2_mm = cond_entropy(seq, 2, corrected=True)
    deltaH_raw = H1_raw - H2_raw
    deltaH_mm = H1_mm - H2_mm
    mm_results[name] = dict(H1_raw=H1_raw, H2_raw=H2_raw, deltaH_raw=deltaH_raw,
                             H1_mm=H1_mm, H2_mm=H2_mm, deltaH_mm=deltaH_mm)
    print(f"{name}: raw deltaH={deltaH_raw:+.4f}   MM-corrected deltaH={deltaH_mm:+.4f}")

print("\n=== Step 2: moving-block bootstrap (L=10, N=500) for deltaH CIs ===")
boot_results = {}
for name in objects:
    seq = labels_per_obj[name]
    rng = np.random.RandomState(42)
    deltas = moving_block_bootstrap(seq, BOOT_BLOCK_LEN, N_BOOTSTRAP, rng)
    deltas = deltas[~np.isnan(deltas)]
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    boot_results[name] = dict(mean=deltas.mean(), std=deltas.std(), ci_low=ci_low, ci_high=ci_high, deltas=deltas)
    print(f"{name}: bootstrap deltaH mean={deltas.mean():+.4f} std={deltas.std():.4f} "
          f"95%CI=[{ci_low:+.4f}, {ci_high:+.4f}]")

print("\n=== Ranking stability check ===")
print("Original point estimates (C2-A): S13=0.0654, S18=0.0735 (highest), S19=0.0176 (lowest)")
print(f"{'Object':>8} {'Boot mean':>12} {'95% CI':>24}")
for name in objects:
    r = boot_results[name]
    print(f"{name:>8} {r['mean']:>+12.4f}   [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

s18_ci = boot_results["S18"]
s19_ci = boot_results["S19"]
overlap = not (s18_ci["ci_low"] > s19_ci["ci_high"] or s19_ci["ci_low"] > s18_ci["ci_high"])
print(f"\nS18 vs S19 95% CI overlap: {overlap} (non-overlap would support a real ranking difference)")

np.savez("cache/C2B_results.npz",
         **{f"{name}_mm_{k}": v for name in objects for k, v in mm_results[name].items()},
         **{f"{name}_boot_deltas": boot_results[name]["deltas"] for name in objects})
