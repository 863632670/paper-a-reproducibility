"""
G2-1B: tau_int and H_s implementations, against G2_0_5_Metric_Definition_Audit.md's
frozen definitions:

tau_int (Paper 63's exact definition): trapezoidal integral of C(t)
  from 0 up to the first negative crossing, evaluated on a LOG-SPACED
  lag grid (not every integer lag). Exact grid points not given in
  Paper 63's text -- documented inference below, not literal
  transcription.

H_s (Paper 77's exact formula): H_s = -sum(p_i * log2(p_i)) / log2(N),
  normalized Shannon entropy of the RAW (unbinned) FFT periodogram,
  excluding the f=0 bin, p_i normalized so sum(p_i)=1 over the
  remaining bins.

C(t) (underlying tau_int): standard normalized magnetization
  autocorrelation, C(t) = <(M_t'-<M>)(M_t'+t-<M>)> / Var(M), matching
  Paper 45's explicit formula.
"""
import numpy as np


def tau_int(macro, max_lag_fraction=0.1, n_log_points=200):
    """
    Paper 63's tau_int: trapezoidal integral of C(t) from 0 to the
    first negative crossing, on a log-spaced lag grid.

    DOCUMENTED INFERENCE (Paper 63 does not give exact grid points):
    log-spaced integer lags from 1 to max_lag_fraction * len(macro),
    deduplicated after rounding to integers (standard practice for
    log-lag autocorrelation to control noise at long lags while
    retaining fine resolution at short lags). max_lag_fraction=0.1
    chosen as a conservative fraction of the ~37,600-sweep trajectory
    length, comfortably covering this system's own established tau
    range (tens to several thousand sweeps) without extending so far
    that noise dominates.
    """
    n = len(macro)
    max_lag = max(int(n * max_lag_fraction), 50)
    log_lags = np.unique(np.round(np.logspace(0, np.log10(max_lag), n_log_points)).astype(int))
    log_lags = log_lags[log_lags >= 1]
    lags = np.concatenate([[0], log_lags])

    m = macro - macro.mean()
    var = np.var(macro)
    C = np.array([1.0] + [np.mean(m[: n - t] * m[t:]) / var for t in log_lags])

    # find first negative crossing
    neg_idx = np.argmax(C < 0) if np.any(C < 0) else len(C)
    if neg_idx == 0:
        return 0.0
    lags_pos = lags[:neg_idx + 1] if neg_idx < len(C) else lags
    C_pos = C[:neg_idx + 1] if neg_idx < len(C) else C
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(C_pos, lags_pos))


def transition_entropy_and_runlength(macro, n_bins=5):
    """
    H_cond = H(M_{t+1}|M_t) and mean run length, on a 5-bin
    discretization of M(t) -- matching Papers 95/96's own method
    ("M(t) discretized into 5 bins; transition matrix P(M_{t+1}|M_t)
    computed; transition entropy H(M_{t+1}|M_t)... and run-length
    statistics (consecutive sweeps remaining in the same bin)").
    Reuses common106.discretize/entropy for the same binning convention
    used throughout this session's own O(B)/TE work.
    """
    import common106 as c
    m_bin, _ = c.discretize(macro, n_bins=n_bins)
    n = len(m_bin)

    # H(M_{t+1}|M_t) = H(M_t, M_{t+1}) - H(M_t)
    joint = np.zeros((n_bins, n_bins))
    for a, b in zip(m_bin[:-1], m_bin[1:]):
        joint[a, b] += 1
    joint_p = joint / joint.sum()
    H_joint = c.entropy(joint_p.flatten())
    H_mt = c.entropy(joint_p.sum(axis=1))
    H_cond = H_joint - H_mt

    # run lengths: consecutive sweeps remaining in the same bin
    runs = []
    cur_len = 1
    for i in range(1, n):
        if m_bin[i] == m_bin[i - 1]:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_len = 1
    runs.append(cur_len)
    mean_run = float(np.mean(runs))
    return float(H_cond), mean_run


def spectral_entropy(macro):
    """
    Paper 77's H_s: normalized Shannon entropy of the raw FFT
    periodogram (mean-subtracted M(t)), excluding f=0, normalized by
    log2(N) so H_s in [0,1].
    """
    m = macro - macro.mean()
    n = len(m)
    S = np.abs(np.fft.rfft(m)) ** 2 / n
    S = S[1:]  # exclude f=0
    p = S / S.sum()
    p = p[p > 0]  # avoid log(0)
    N = len(p)
    H = -np.sum(p * np.log2(p))
    return float(H / np.log2(N))
