"""
R100-B: Paper 102 reproduction ("The Compression Test: Full Structure
Does Not Predict O500 Either"). LOO ridge regression comparing a
baseline model (log(tau), H_s -- 2 features) against a full-structure
model (50 features: 25 raw transition-matrix P(Mt+1|Mt) entries + 25
features measuring how much each transition probability varies across
the 4 possible (S_t,S_t-1) source-history conditions), predicting O500,
n=24.

Paper 102's own reported values:
  Baseline LOO: r=-0.173, p=0.418
  Full model, light regularization (alpha=1,10): r~0.00 to -0.02 (worse than baseline)
  Full model, heavy regularization (alpha=50): r=0.182, p=0.394

NOTE on the 25 variance features: paper's Method specifies "variance of
that transition probability across the 4 possible (S_t,S_t-1) source-
history conditions" but does not specify how missing/zero-count
(Mt, condition) cells are handled. This reproduction treats a missing
condition's row as an all-zero distribution (a documented inference,
flagged here -- analogous to the tau_int log-lag-grid gap already
flagged in G2-0.5).
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import common106 as c
from metrics_tau_hs import tau_int, spectral_entropy

seeds = list(range(180000, 180024))
n_bins = 5
conditions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def full_features(source, macro):
    m_bin, _ = c.discretize(macro, n_bins=n_bins)
    s = source
    n = len(m_bin)

    Mt = m_bin[:-1]
    Mt1 = m_bin[1:]
    counts = np.zeros((n_bins, n_bins))
    for a, b in zip(Mt, Mt1):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    feat_topo = P.flatten()

    idx_range = np.arange(1, n - 1)
    Mt_c = m_bin[idx_range]
    Mt1_c = m_bin[idx_range + 1]
    St_c = s[idx_range]
    Stm1_c = s[idx_range - 1]

    cond_P = np.zeros((4, n_bins, n_bins))
    for ci, (a, b) in enumerate(conditions):
        mask = (St_c == a) & (Stm1_c == b)
        for mt in range(n_bins):
            mmask = mask & (Mt_c == mt)
            cnt = mmask.sum()
            if cnt > 0:
                dist = np.bincount(Mt1_c[mmask], minlength=n_bins) / cnt
                cond_P[ci, mt, :] = dist
    var_feat = np.var(cond_P, axis=0)
    feat_var = var_feat.flatten()

    return np.concatenate([feat_topo, feat_var])


print("computing features for all 24 corrected trajectories...")
baseline_X = []
full_X = []
o500 = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    t = tau_int(macro)
    h = spectral_entropy(macro)
    baseline_X.append([np.log(t), h])
    full_X.append(full_features(source, macro))
    o500.append(c.O_of_B(source, macro, 500, rng_seed=0))

baseline_X = np.array(baseline_X)
full_X = np.array(full_X)
o500 = np.array(o500)
n = len(seeds)


def loo_ridge(X, y, alpha):
    preds = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[i:i + 1])
        model = Ridge(alpha=alpha)
        model.fit(Xtr, y[train_idx])
        preds[i] = model.predict(Xte)[0]
    r, p = stats.pearsonr(preds, y)
    return r, p


print("\n=== Baseline model (log(tau), H_s), alpha=1 ===")
r_b, p_b = loo_ridge(baseline_X, o500, alpha=1.0)
print(f"LOO r={r_b:.3f} p={p_b:.3f}   (paper: r=-0.173, p=0.418)   [Level-2 caveat: tau_int]")

print("\n=== Full model (50 features: transition matrix + source-history variance) ===")
for alpha in [1, 10, 50]:
    r_f, p_f = loo_ridge(full_X, o500, alpha=alpha)
    print(f"alpha={alpha:3d}: LOO r={r_f:.3f} p={p_f:.3f}")
print("(paper: alpha=1,10 -> r~0.00 to -0.02 (worse than baseline); alpha=50 -> r=0.182, p=0.394)")
