"""
R100-A: Paper 99 reproduction ("The Fifth Candidate Also Fails:
Single-Step Source-State Coupling Does Not Predict O500"). Tests
A_S = <|P(M_t+1|M_t,S_t=+1) - P(M_t+1|M_t,S_t=-1)|> and
dH_S = H(M_t+1|M_t) - H(M_t+1|M_t,S_t) vs O500, n=24, plus tau-partial
correlation and the golden-pair (180013/180009) direct comparison.

Paper 99's own reported values:
  A_S vs O500: r=0.212, p=0.321
  log(dH_S) vs O500: r=-0.227, p=0.286
  partial (control tau): A_S|tau: r=0.151, p=0.481; log(dH_S)|tau: r=-0.181, p=0.398
  Golden pair A_S: 180013=0.0267, 180009=0.0301 (13% diff)
  Golden pair dH_S: 180013=0.00018, 180009=0.00089 (~5-fold, low-O500 higher)
"""
import numpy as np
from scipy import stats
from collections import Counter
import common106 as c
from metrics_tau_hs import tau_int

seeds = list(range(180000, 180024))


def A_S_and_dHS(source, macro, n_bins=5, min_samples=10):
    m_bin, _ = c.discretize(macro, n_bins=n_bins)
    s = source
    Mt = m_bin[:-1]
    Mt1 = m_bin[1:]
    St = s[:-1]
    n = len(Mt)

    diffs = []
    for mb in range(n_bins):
        mask = Mt == mb
        mask_pos = mask & (St == 1)
        mask_neg = mask & (St == -1)
        if mask_pos.sum() >= min_samples and mask_neg.sum() >= min_samples:
            p_pos = np.bincount(Mt1[mask_pos], minlength=n_bins) / mask_pos.sum()
            p_neg = np.bincount(Mt1[mask_neg], minlength=n_bins) / mask_neg.sum()
            diffs.append(np.sum(np.abs(p_pos - p_neg)))
    A_S = float(np.mean(diffs)) if diffs else np.nan

    def H_of(counter, total):
        return -sum((v / total) * np.log2(v / total) for v in counter.values())

    joint_Mt1Mt = Counter(zip(Mt.tolist(), Mt1.tolist()))
    hist_Mt = Counter(Mt.tolist())
    H_cond1 = H_of(joint_Mt1Mt, n) - H_of(hist_Mt, n)

    joint_triple = Counter(zip(Mt.tolist(), St.tolist(), Mt1.tolist()))
    joint_pair = Counter(zip(Mt.tolist(), St.tolist()))
    H_cond2 = H_of(joint_triple, n) - H_of(joint_pair, n)

    dH_S = H_cond1 - H_cond2
    return A_S, dH_S


def partial_corr(x, y, z):
    bx = np.polyfit(z, x, 1)
    by = np.polyfit(z, y, 1)
    rx = x - np.polyval(bx, z)
    ry = y - np.polyval(by, z)
    return stats.pearsonr(rx, ry)


print("computing A_S, dH_S, tau, O500 for all 24 corrected trajectories...")
rows = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    A_S, dH_S = A_S_and_dHS(source, macro)
    t = tau_int(macro)
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)
    rows.append((seed, A_S, dH_S, t, o500))

seeds_arr = np.array([r[0] for r in rows])
A_S_arr = np.array([r[1] for r in rows])
dH_S_arr = np.array([r[2] for r in rows])
taus = np.array([r[3] for r in rows])
o500 = np.array([r[4] for r in rows])

print("\n=== Paper 99 core test ===")
r1, p1 = stats.pearsonr(A_S_arr, o500)
print(f"A_S vs O500:        r={r1:.3f} p={p1:.3f}   (paper: r=0.212, p=0.321)")
r2, p2 = stats.pearsonr(np.log(dH_S_arr), o500)
print(f"log(dH_S) vs O500:  r={r2:.3f} p={p2:.3f}   (paper: r=-0.227, p=0.286)")

print("\n=== Partial correlations, controlling log(tau) ===")
logtau = np.log(taus)
rp1, pp1 = partial_corr(A_S_arr, o500, logtau)
print(f"A_S|tau:       r={rp1:.3f} p={pp1:.3f}   (paper: r=0.151, p=0.481)")
rp2, pp2 = partial_corr(np.log(dH_S_arr), o500, logtau)
print(f"log(dH_S)|tau: r={rp2:.3f} p={pp2:.3f}   (paper: r=-0.181, p=0.398)")

print("\n=== Golden pair direct comparison ===")
idx13 = seeds.index(180013)
idx09 = seeds.index(180009)
print(f"180013 (high-O500): A_S={A_S_arr[idx13]:.4f}  dH_S={dH_S_arr[idx13]:.5f}")
print(f"180009 (low-O500):  A_S={A_S_arr[idx09]:.4f}  dH_S={dH_S_arr[idx09]:.5f}")
print(f"(paper: A_S 0.0267 vs 0.0301 [13% diff]; dH_S 0.00018 vs 0.00089 [~5-fold])")
pct_diff = abs(A_S_arr[idx13] - A_S_arr[idx09]) / max(A_S_arr[idx13], A_S_arr[idx09]) * 100
print(f"This reproduction A_S %% diff: {pct_diff:.1f}%%")
