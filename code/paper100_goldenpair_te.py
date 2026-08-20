"""
Reconstruction of Paper 100's golden-pair result (P3-A1 Phase 3).

Uses TE_k (from paper101e_regime_mixing.py, verified on synthetic data
in Paper100_TE_Implementation_Check.md), NOT common106.compute_TE
(a different, O500-purposed metric) -- see Paper100_Code_Provenance_Check.md
for why. subset_mask=None (whole-trajectory, matching Paper 100's own
design -- no regime-splitting, unlike 101E's stable/flip variant).

Paper's reported values:
  High-O500 (180013): O500=1.993  TE1=0.000181  TE2/TE1=5.716
  Low-O500  (180009): O500=-2.065 TE1=0.000888  TE2/TE1=1.688
"""
import numpy as np
import common106 as c


def discretize_pair(source, macro, macro_bins=5):
    s_bin = (source > 0).astype(int)
    m_bin, _ = c.discretize(macro, n_bins=macro_bins)
    return s_bin, m_bin


def TE_k(s_bin, m_bin, k, subset_mask=None):
    n = len(m_bin)
    idx = np.arange(1, n - 1)
    if subset_mask is not None:
        idx = idx[subset_mask]
    Mt = m_bin[idx]
    Mt1 = m_bin[idx + 1]
    St = s_bin[idx]
    if k == 1:
        joint_MtMt1 = np.zeros((5, 5))
        for a, b in zip(Mt, Mt1):
            joint_MtMt1[a, b] += 1
        p1 = joint_MtMt1 / joint_MtMt1.sum()
        H_Mt1_given_Mt = c.entropy(p1.flatten()) - c.entropy(p1.sum(axis=1))

        joint_MtStMt1 = np.zeros((5, 2, 5))
        for a, s, b in zip(Mt, St, Mt1):
            joint_MtStMt1[a, s, b] += 1
        p2 = joint_MtStMt1 / joint_MtStMt1.sum()
        H_MtSt = c.entropy(p2.sum(axis=2).flatten())
        H_MtStMt1 = c.entropy(p2.flatten())
        H_Mt1_given_MtSt = H_MtStMt1 - H_MtSt
        return H_Mt1_given_Mt - H_Mt1_given_MtSt
    elif k == 2:
        Stm1 = s_bin[idx - 1]
        joint_MtMt1 = np.zeros((5, 5))
        for a, b in zip(Mt, Mt1):
            joint_MtMt1[a, b] += 1
        p1 = joint_MtMt1 / joint_MtMt1.sum()
        H_Mt1_given_Mt = c.entropy(p1.flatten()) - c.entropy(p1.sum(axis=1))

        joint = np.zeros((5, 2, 2, 5))
        for a, s0, s1, b in zip(Mt, St, Stm1, Mt1):
            joint[a, s0, s1, b] += 1
        p2 = joint / joint.sum()
        H_cond = c.entropy(p2.sum(axis=3).flatten())
        H_full = c.entropy(p2.flatten())
        H_Mt1_given_cond = H_full - H_cond
        return H_Mt1_given_Mt - H_Mt1_given_cond


def analyze(source, macro, label, paper_O500, paper_TE1, paper_ratio):
    s_bin, m_bin = discretize_pair(source, macro)
    te1 = TE_k(s_bin, m_bin, 1)
    te2 = TE_k(s_bin, m_bin, 2)
    ratio = te2 / te1 if te1 else float("nan")
    o500 = c.O_of_B(source, macro, 500, rng_seed=0)

    print(f"\n{label}:")
    print(f"  O500 (this reconstruction) = {o500:.3f}   (paper: {paper_O500})")
    print(f"  TE1  (this reconstruction) = {te1:.6f}   (paper: {paper_TE1})")
    print(f"  TE2  (this reconstruction) = {te2:.6f}")
    print(f"  TE2/TE1 (this reconstruction) = {ratio:.3f}   (paper: {paper_ratio})")
    return o500, te1, te2, ratio


d13 = np.load("cache/s13_m13.npz")
source13, macro13 = d13["source13"], d13["macro13"]
d09 = np.load("cache/s09_traj.npz")
source09, macro09 = d09["source"], d09["macro"]

r13 = analyze(source13, macro13, "High-O500 (180013)", 1.993, 0.000181, 5.716)
r09 = analyze(source09, macro09, "Low-O500 (180009)", -2.065, 0.000888, 1.688)

print("\n=== Direction check ===")
print(f"180013 TE2/TE1 ({r13[3]:.3f}) > 180009 TE2/TE1 ({r09[3]:.3f}) ? {r13[3] > r09[3]}")
print(f"180013 O500 ({r13[0]:.3f}) > 180009 O500 ({r09[0]:.3f}) ? {r13[0] > r09[0]}")
