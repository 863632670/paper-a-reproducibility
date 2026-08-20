"""
R100-A: Paper 96 reproduction ("The Gate Does Not Pass: Transition
Persistence Does Not Generalize"). Tests H_cond, log(mean run length),
log(max run length) vs O500, simple + tau-partial correlation.

Paper 96's own reported values:
  H_cond vs O500: r=0.352, p=0.091
  log(mean run) vs O500: r=-0.265, p=0.210
  log(max run) vs O500: r=-0.297, p=0.159
  (partial, controlling log(tau)): r=0.239/0.026/-0.078
"""
import numpy as np
from scipy import stats
import common106 as c
from metrics_tau_hs import tau_int, transition_entropy_and_runlength

seeds = list(range(180000, 180024))


def max_run_length(macro, n_bins=5):
    m_bin, _ = c.discretize(macro, n_bins=n_bins)
    runs = []
    cur = 1
    for i in range(1, len(m_bin)):
        if m_bin[i] == m_bin[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return max(runs)


print("computing H_cond, mean/max run length, tau, O500 for all 24 corrected trajectories...")
rows = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    h_cond, mean_run = transition_entropy_and_runlength(macro)
    max_run = max_run_length(macro)
    t = tau_int(macro)
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)
    rows.append((seed, h_cond, mean_run, max_run, t, o500))

seeds_arr = np.array([r[0] for r in rows])
hcond = np.array([r[1] for r in rows])
meanrun = np.array([r[2] for r in rows])
maxrun = np.array([r[3] for r in rows])
taus = np.array([r[4] for r in rows])
o500 = np.array([r[5] for r in rows])


def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z, via residualization."""
    bx = np.polyfit(z, x, 1)
    by = np.polyfit(z, y, 1)
    rx = x - np.polyval(bx, z)
    ry = y - np.polyval(by, z)
    return stats.pearsonr(rx, ry)


print("\n=== Paper 96 core test ===")
r1, p1 = stats.pearsonr(hcond, o500)
print(f"H_cond vs O500:         r={r1:.3f} p={p1:.3f}   (paper: r=0.352, p=0.091)")
r2, p2 = stats.pearsonr(np.log(meanrun), o500)
print(f"log(mean run) vs O500:  r={r2:.3f} p={p2:.3f}   (paper: r=-0.265, p=0.210)")
r3, p3 = stats.pearsonr(np.log(maxrun), o500)
print(f"log(max run) vs O500:   r={r3:.3f} p={p3:.3f}   (paper: r=-0.297, p=0.159)")

print("\n=== Partial correlations, controlling log(tau) [caveat: tau_int implementation] ===")
logtau = np.log(taus)
rp1, pp1 = partial_corr(hcond, o500, logtau)
print(f"H_cond|tau:        r={rp1:.3f} p={pp1:.3f}   (paper: r=0.239, p=0.260)")
rp2, pp2 = partial_corr(np.log(meanrun), o500, logtau)
print(f"log(mean run)|tau: r={rp2:.3f} p={pp2:.3f}   (paper: r=0.026, p=0.904)")
rp3, pp3 = partial_corr(np.log(maxrun), o500, logtau)
print(f"log(max run)|tau:  r={rp3:.3f} p={pp3:.3f}   (paper: r=-0.078, p=0.716)")

print("\n=== Sign check vs golden pair (paper's own highlighted finding) ===")
idx13 = seeds.index(180013)
idx09 = seeds.index(180009)
print(f"180013 (high-O500): mean_run={meanrun[idx13]:.1f}  H_cond={hcond[idx13]:.4f}")
print(f"180009 (low-O500):  mean_run={meanrun[idx09]:.1f}  H_cond={hcond[idx09]:.4f}")
print(f"(paper: high-O500 had LONGER mean run (287) & LOWER H_cond (0.029) than low-O500 (137, 0.056) --")
print(f" but population-wide trend was OPPOSITE direction, the paper's own headline surprise)")
