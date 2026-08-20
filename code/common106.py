"""
Shared reconstructed infrastructure for the 106-108 ISRC series.

PROVENANCE NOTE: run_trial_combined / discretize / entropy / compute_TE /
exp_A_order_shuffle / exp_B_internal_shuffle / O_of_B / random_eulerian_rearrangement
(k=2 case) / second_order_kernel_counts are transcribed verbatim (or with only
mechanical generalization, noted per-function) from
transcripts/2026-08-13-09-47-34-isrc-papers-107-165-ising-grammar-selection.txt
(lines ~17167+, the code cell that produced Paper 106-U). This is the actual
code executed in the original session for the 106-A..106-Z range.

Everything from 106-AA onward (including G3/G4/G5 grammar order, and all of
the AB/AC/107/108 series) is NOT present in any surviving transcript -- no
transcript file mentions "106-AA", "106-AB", "106-AC", or "108" anywhere.
generalized_eulerian_rearrangement's k>2 behavior is therefore a faithful
MECHANICAL GENERALIZATION of the verified k=2 code (same de Bruijn multigraph
+ randomized Hierholzer's algorithm construction, generalized from 2-symbol to
k-symbol history states), cross-checked against Paper 106-Y's Method section
("The 3rd-order de Bruijn-style transition multigraph built from S13's true
full trajectory: nodes representing 3-symbol history states, edges
representing (history)->(next-symbol) transitions with exact multiplicity.
... n=50 independent Eulerian paths generated ... via randomized Hierholzer's
algorithm (seeds 1-50)") -- NOT itself extracted from surviving code. Treat
k>2 results as Category-C (reimplemented-from-text) confidence, not
Category-B (original-code) confidence, even though it shares the k=2
implementation.

Trajectory identity: "S13"/"M13" = the true source/macro series of the
trajectory generated with seed=180013 (confirmed via Paper 100's golden-pair
table: "High-O500 (180013)"; Paper 106D1's title "180013_Stands_Alone").
"""

import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paper33_size_scaling import make_masks, checkerboard_sweep  # noqa: E402

L = 96
R_SOURCE = 6
T_THERMAL_2_25 = 4700


def run_trial_combined(seed, T=2.25, t_thermal=T_THERMAL_2_25, therm_mult=2.0, prod_mult=8.0):
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


def discretize(series, n_bins=5):
    edges = np.linspace(series.min() - 1e-9, series.max() + 1e-9, n_bins + 1)
    return np.clip(np.digitize(series, edges) - 1, 0, n_bins - 1), edges


def entropy(p):
    p_nz = p[p > 0]
    return float(-np.sum(p_nz * np.log2(p_nz)))


def compute_TE(source, macro, dt=1000, source_bins=2, macro_bins=5):
    n = len(macro)
    if n - dt < 30:
        return None
    s_bins_full, _ = discretize(source, source_bins)
    m_bins_full, _ = discretize(macro, macro_bins)
    s_t = s_bins_full[:n - dt]
    m_t = m_bins_full[:n - dt]
    m_future = m_bins_full[dt:]
    joint2 = np.zeros((macro_bins, macro_bins))
    for a, b in zip(m_t, m_future):
        joint2[a, b] += 1
    joint2_p = joint2 / joint2.sum()
    Hmt = entropy(joint2_p.sum(axis=1))
    Hmm = entropy(joint2_p.flatten())
    U = Hmm - Hmt
    joint3 = np.zeros((macro_bins, source_bins, macro_bins))
    for a, c, b in zip(m_t, s_t, m_future):
        joint3[a, c, b] += 1
    joint3_p = joint3 / joint3.sum()
    joint_sm_p = joint3_p.sum(axis=2)
    H_sm = entropy(joint_sm_p.flatten())
    H_smm = entropy(joint3_p.flatten())
    H_mfuture_given_mt_s = H_smm - H_sm
    TE = U - H_mfuture_given_mt_s
    return TE


def exp_A_order_shuffle(source, macro, B, rng_seed=0):
    n = len(macro)
    n_blocks = n // B
    rng = np.random.RandomState(rng_seed)
    block_order = rng.permutation(n_blocks)
    source_shuf = np.concatenate([source[i * B:(i + 1) * B] for i in block_order])
    macro_shuf = np.concatenate([macro[i * B:(i + 1) * B] for i in block_order])
    return source_shuf, macro_shuf


def exp_B_internal_shuffle(source, macro, B, rng_seed=0):
    n = len(macro)
    n_blocks = n // B
    rng = np.random.RandomState(rng_seed)
    source_shuf = source[:n_blocks * B].copy()
    macro_shuf = macro[:n_blocks * B].copy()
    for i in range(n_blocks):
        perm = rng.permutation(B)
        source_shuf[i * B:(i + 1) * B] = source_shuf[i * B:(i + 1) * B][perm]
        macro_shuf[i * B:(i + 1) * B] = macro_shuf[i * B:(i + 1) * B][perm]
    return source_shuf, macro_shuf


def O_of_B(source, macro, B, rng_seed=0):
    sA, mA = exp_A_order_shuffle(source, macro, B, rng_seed)
    TE_A = compute_TE(sA, mA)
    sB, mB = exp_B_internal_shuffle(source, macro, B, rng_seed)
    TE_B = compute_TE(sB, mB)
    if TE_A and TE_B and TE_A > 0 and TE_B > 0:
        return float(np.log(TE_A / TE_B))
    return None


def _hierholzer(start, edge_dict):
    local = {k: list(v) for k, v in edge_dict.items()}
    node_stack = [start]
    result_nodes = []
    result_symbols = []
    while node_stack:
        v = node_stack[-1]
        if local[v]:
            dst, sym = local[v].pop()
            node_stack.append(dst)
            result_symbols.append(sym)
        else:
            result_nodes.append(node_stack.pop())
    result_nodes.reverse()
    result_symbols.reverse()
    return result_nodes, result_symbols


def random_eulerian_rearrangement(segment, rng_seed=0):
    """k=2 (G2) case, transcribed verbatim from the surviving transcript."""
    return generalized_eulerian_rearrangement(segment, order=2, rng_seed=rng_seed)


def generalized_eulerian_rearrangement(segment, order, rng_seed=0):
    """
    Gk-preserving Eulerian reconstruction: build the de Bruijn-style transition
    multigraph on `order`-symbol history states (nodes) with edges = each
    observed (history)->(next value) transition, multiplicity = exact count.
    Randomized Hierholzer's algorithm finds a different Eulerian path through
    the SAME multigraph, preserving exact k-th order transition statistics
    while randomizing the specific realized path.

    k=2 body verified against the surviving transcript's
    `random_eulerian_rearrangement`. k>2 is a mechanical generalization
    (see module docstring) cross-checked against Paper 106-Y's Method section.
    """
    import itertools

    s = (segment > 0).astype(int)
    n = len(s)
    rng = np.random.RandomState(rng_seed)

    def make_node(t):
        return tuple(int(x) for x in s[t - order:t])

    # Pre-populate all 2^order possible history nodes in a fixed canonical
    # order, matching the transcript's k=2 code (which pre-populates
    # [(0,0),(0,1),(1,0),(1,1)] before adding edges) -- this keeps dict
    # insertion order (and therefore rng.shuffle consumption order)
    # independent of which nodes happen to appear first while scanning t.
    edges = {node: [] for node in itertools.product([0, 1], repeat=order)}
    for t in range(order, n):
        src = make_node(t)
        symbol = int(s[t])
        dst = tuple(list(src[1:]) + [symbol])
        edges[src].append((dst, symbol))

    start_node = tuple(int(x) for x in s[0:order])

    edge_copy = {node: list(lst) for node, lst in edges.items()}
    for node in edge_copy:
        rng.shuffle(edge_copy[node])

    result_nodes, result_symbols = _hierholzer(start_node, edge_copy)

    new_seq = list(start_node) + result_symbols
    new_seq = np.array(new_seq[:n])
    return np.where(new_seq == 1, 1.0, -1.0)


def kth_order_kernel_counts(segment, order):
    s = (segment > 0).astype(int)
    n = len(s)
    counts = {}
    for t in range(order, n):
        hist = tuple(s[t - order:t])
        counts.setdefault(hist, [0, 0])
        counts[hist][s[t]] += 1
    return counts


def kth_order_entropy_gain(segment, order):
    """H(order-1) - H(order): the extra predictive gain from one more symbol
    of history, matching the 'entropy gain' quantity Paper 106-Y verifies
    (0.0283 for true S13 at order=3)."""
    def cond_entropy(k):
        counts = kth_order_kernel_counts(segment, k)
        H = 0.0
        total_all = sum(sum(v) for v in counts.values())
        for hist, (c0, c1) in counts.items():
            tot = c0 + c1
            if tot == 0:
                continue
            p0, p1 = c0 / tot, c1 / tot
            h = 0.0
            if p0 > 0:
                h -= p0 * np.log2(p0)
            if p1 > 0:
                h -= p1 * np.log2(p1)
            H += (tot / total_all) * h
        return H

    return cond_entropy(order - 1) - cond_entropy(order)


def block_majority_sequence(series, w):
    s = (np.asarray(series) > 0).astype(int)
    n_blocks = len(s) // w
    blocks = s[:n_blocks * w].reshape(n_blocks, w)
    majority = (blocks.sum(axis=1) > (w / 2.0)).astype(int)
    return majority


def block_transition_entropy(series, w):
    b = block_majority_sequence(series, w)
    if len(b) < 2:
        return float("nan")
    counts = {0: [0, 0], 1: [0, 0]}
    for a, nxt in zip(b[:-1], b[1:]):
        counts[a][nxt] += 1
    H = 0.0
    total_all = sum(sum(v) for v in counts.values())
    if total_all == 0:
        return float("nan")
    for prev, (c0, c1) in counts.items():
        tot = c0 + c1
        if tot == 0:
            continue
        p0, p1 = c0 / tot, c1 / tot
        h = 0.0
        if p0 > 0:
            h -= p0 * np.log2(p0)
        if p1 > 0:
            h -= p1 * np.log2(p1)
        H += (tot / total_all) * h
    return H
