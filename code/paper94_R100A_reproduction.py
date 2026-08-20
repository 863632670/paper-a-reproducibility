"""
R100-A: Paper 94 reproduction ("Decisive: Tau Does Not Predict Ordering
Strength"). Tests log(tau) vs O500 and H_s vs O500 across all 24
corrected trajectories.

KNOWN CAVEAT, flagged before running: this session's tau_int
implementation (G2_0_5) is a documented INFERENCE for the exact
log-lag grid/cutoff (Paper 63 does not specify exact grid points).
Spot-check against Paper 94's own reported values for the golden pair
(180009: tau=3980.8; 180013: tau=5997.6) shows this session's tau_int
values (2972.4, 3375.4) are systematically LOWER -- almost certainly
because this session's max_lag_fraction=0.1 (~3760 sweeps) cuts off
the integral before slow trajectories' full decay tail. H_s carries no
such ambiguity (exact formula, already validated). tau-based results
below should be read as Level 2 (directional) at best; H_s-based
results carry the full weight of exact-formula reproduction.

Paper 94's own reported values:
  log(tau) vs O500: r=-0.289, p=0.170
  H_s vs O500: r=0.262, p=0.216
"""
import numpy as np
from scipy import stats
import common106 as c
from metrics_tau_hs import tau_int, spectral_entropy

seeds = list(range(180000, 180024))

print("computing tau_int, H_s, O500 for all 24 corrected trajectories...")
rows = []
for seed in seeds:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    t = tau_int(macro)
    h = spectral_entropy(macro)
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)
    rows.append((seed, t, h, o500))

seeds_arr = np.array([r[0] for r in rows])
taus = np.array([r[1] for r in rows])
hs = np.array([r[2] for r in rows])
o500 = np.array([r[3] for r in rows])

print("\n=== Paper 94 core test ===")
r_tau, p_tau = stats.pearsonr(np.log(taus), o500)
print(f"log(tau) vs O500: r={r_tau:.3f} p={p_tau:.3f}   (paper: r=-0.289, p=0.170)  [Level-2 caveat: tau_int implementation]")
r_hs, p_hs = stats.pearsonr(hs, o500)
print(f"H_s vs O500:      r={r_hs:.3f} p={p_hs:.3f}   (paper: r=0.262, p=0.216)  [exact-formula H_s]")

print("\n=== Massive variance within similar-tau trajectories (paper's Table 3.2 style check) ===")
order = np.argsort(taus)
print("Sorted by tau_int (this reproduction):")
for i in order:
    print(f"  seed {seeds_arr[i]}: tau_int={taus[i]:.1f}  O500={o500[i]:.3f}")

np.savez("cache/R100A_paper94_results.npz", seeds=seeds_arr, tau_int=taus, H_s=hs, O500=o500)
