"""
Paper 33: Size scaling of the synergy peak -- testing Delta_t_peak(L) ~ L^z.

Repeats Paper 31/32's PID decomposition at T_c=2.269, fixed source r=1, across
multiple system sizes L, to test whether the synergy peak's time location scales
with L as a power law, which would connect ISRC's synergy timescale to a
conventional dynamic critical exponent.

Run one L at a time via run_L(L, t_thermal, n_seeds).
"""

import numpy as np


T_C = 2.269
R_SOURCE = 1
DT_VALUES = [10, 50, 100, 200, 400, 800]


def make_masks(L):
    I, J = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    mask_a = ((I + J) % 2 == 0)
    return mask_a, ~mask_a


def checkerboard_sweep(spins, beta, rng, mask_a, mask_b):
    for mask in (mask_a, mask_b):
        neighbors = (np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0) +
                     np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1))
        dE = 2 * spins * neighbors
        accept_prob = np.where(dE <= 0, 1.0, np.exp(-beta * dE))
        rand = rng.random(size=spins.shape)
        flip = mask & (rand < accept_prob)
        spins[flip] *= -1


def run_trial(seed, L, t_thermal, T=T_C, therm_mult=2.0, prod_mult=8.0):
    therm = int(therm_mult * t_thermal)
    prod = int(prod_mult * t_thermal)
    rng = np.random.RandomState(seed)
    beta = 1.0 / T
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    mask_a, mask_b = make_masks(L)
    for _ in range(therm):
        checkerboard_sweep(spins, beta, rng, mask_a, mask_b)
    source_series, macro_series = [], []
    for _ in range(prod):
        checkerboard_sweep(spins, beta, rng, mask_a, mask_b)
        source_series.append(float(spins[R_SOURCE, 0]))
        macro_series.append(float(np.sum(spins)))
    return np.array(source_series), np.array(macro_series)


def discretize(series, n_bins):
    edges = np.linspace(series.min() - 1e-9, series.max() + 1e-9, n_bins + 1)
    return np.clip(np.digitize(series, edges) - 1, 0, n_bins - 1)


def entropy(p):
    p_nz = p[p > 0]
    return float(-np.sum(p_nz * np.log2(p_nz)))


def mi_2var(x_series, y_series, dt, x_bins, y_bins):
    if dt >= len(y_series):
        return float("nan")
    x_d = discretize(x_series[:-dt], x_bins)
    y_d = discretize(y_series[dt:], y_bins)
    joint = np.zeros((x_bins, y_bins))
    for a, b in zip(x_d, y_d):
        joint[a, b] += 1
    joint_p = joint / joint.sum()
    Hx = entropy(joint_p.sum(axis=1))
    Hy = entropy(joint_p.sum(axis=0))
    Hxy = entropy(joint_p.flatten())
    return Hx + Hy - Hxy


def conditional_te(source_series, macro_series, dt, source_bins, macro_bins):
    if dt >= len(macro_series):
        return float("nan")
    s_bins = discretize(source_series[:-dt], source_bins)
    m_t_bins = discretize(macro_series[:-dt], macro_bins)
    m_future_bins = discretize(macro_series[dt:], macro_bins)
    joint_mm = np.zeros((macro_bins, macro_bins))
    for a, b in zip(m_t_bins, m_future_bins):
        joint_mm[a, b] += 1
    joint_mm_p = joint_mm / joint_mm.sum()
    H_mt = entropy(joint_mm_p.sum(axis=1))
    H_mm = entropy(joint_mm_p.flatten())
    H_mfuture_given_mt = H_mm - H_mt
    joint_smm = np.zeros((macro_bins, source_bins, macro_bins))
    for a, c, b in zip(m_t_bins, s_bins, m_future_bins):
        joint_smm[a, c, b] += 1
    joint_smm_p = joint_smm / joint_smm.sum()
    joint_sm_p = joint_smm_p.sum(axis=2)
    H_sm = entropy(joint_sm_p.flatten())
    H_smm = entropy(joint_smm_p.flatten())
    H_mfuture_given_mt_s = H_smm - H_sm
    return H_mfuture_given_mt - H_mfuture_given_mt_s


def pid_syn(I1, I2, TE):
    I_joint = I2 + TE
    return I_joint - max(I1, I2)


def run_L(L, t_thermal, n_seeds=12):
    print(f"=== L={L}, t_thermal={t_thermal}, {n_seeds} seeds ===")
    syn_by_dt = {dt: [] for dt in DT_VALUES}
    for seed in range(n_seeds):
        source, macro = run_trial(seed + 22000, L, t_thermal)
        for dt in DT_VALUES:
            I1 = mi_2var(source, macro, dt, 2, 5)
            I2 = mi_2var(macro, macro, dt, 5, 5)
            TE = conditional_te(source, macro, dt, 2, 5)
            syn_by_dt[dt].append(pid_syn(I1, I2, TE))
        print(f"  seed {seed} done")
    means = {dt: float(np.mean(syn_by_dt[dt])) for dt in DT_VALUES}
    peak_dt = max(means, key=means.get)
    print(f"L={L} Syn(dt): {means}")
    print(f"L={L} peak at dt={peak_dt}")
    return means, peak_dt
