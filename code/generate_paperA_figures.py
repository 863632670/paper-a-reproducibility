"""
Generates Paper A's five manuscript figures from already-cached
reproduction data. No new simulation -- pure plotting from existing,
independently-verified numeric results (G2_1B2, G2_2, G2_3, plus small
schematic diagrams for Figures 1 and 4).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.cluster import AgglomerativeClustering

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
OUT = "../figures"

GOLD13 = "#c0392b"
GOLD09 = "#2980b9"
GREY = "#7f8c8d"

# ============================================================
# Figure 1: the problem, in two rows -- traditional description
# collapses distinct trajectories; this paper's representation
# keeps them separable. Designed for ~10-second comprehension.
# ============================================================
rng_fig1 = np.random.default_rng(3)


def squiggle(ax, x0, y0, w, h, color, seed):
    r = np.random.default_rng(seed)
    xs = np.linspace(0, w, 60)
    ys = np.cumsum(r.normal(0, 1, 60))
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9) * h
    ax.plot(x0 + xs, y0 + ys, color=color, linewidth=1.8, solid_capstyle="round")


fig, axes = plt.subplots(2, 1, figsize=(11, 6.4))

row_specs = [
    ("Traditional description", "scalar summary", True),
    ("This paper's representation", "trajectory-level\nrepresentation", False),
]

for row, (row_title, box_label, collapses) in enumerate(row_specs):
    ax = axes[row]
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.3, 3.5)
    ax.axis("off")

    # two distinct trajectories on the left
    squiggle(ax, 0.3, 1.55, 1.6, 0.95, GOLD13, seed=13)
    squiggle(ax, 0.3, 0.15, 1.6, 0.95, GOLD09, seed=9)
    ax.text(1.1, 2.68, "trajectory A", ha="center", fontsize=8.5, color=GOLD13)
    ax.text(1.1, -0.18, "trajectory B", ha="center", fontsize=8.5, color=GOLD09)

    ax.annotate("", xy=(3.0, 1.5), xytext=(2.1, 1.5), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))

    box = FancyBboxPatch((3.1, 1.0), 2.3, 1.0, boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor="#333333", facecolor="#ecf0f1")
    ax.add_patch(box)
    ax.text(4.25, 1.5, box_label, ha="center", va="center", fontsize=10, fontweight="bold")

    ax.annotate("", xy=(5.9, 1.5), xytext=(5.5, 1.5), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))

    if collapses:
        outbox = FancyBboxPatch((6.0, 1.15), 2.0, 0.7, boxstyle="round,pad=0.05,rounding_size=0.08",
                                 linewidth=1.8, edgecolor="#c0392b", facecolor="#c0392b", alpha=0.15)
        ax.add_patch(outbox)
        ax.text(7.0, 1.5, "same value", ha="center", va="center", fontsize=10, fontweight="bold", color="#c0392b")
        ax.text(8.6, 1.5, "A and B become\nindistinguishable", ha="left", va="center", fontsize=9, color="#c0392b")
    else:
        ax.scatter([6.6], [1.85], marker="*", s=350, color=GOLD13, edgecolor="black", linewidth=1, zorder=5)
        ax.scatter([6.6], [1.15], marker="*", s=350, color=GOLD09, edgecolor="black", linewidth=1, zorder=5)
        ax.plot([6.3, 8.9], [1.5, 1.5], color="#27ae60", linewidth=2, alpha=0.5, zorder=1)
        ax.text(9.1, 1.5, "A and B remain\ndistinguishable", ha="left", va="center", fontsize=9, color="#27ae60", fontweight="bold")

    ax.text(0.0, 3.25, row_title, ha="left", fontsize=11.5, fontweight="bold",
            color="#c0392b" if collapses else "#27ae60")

plt.subplots_adjust(hspace=0.55)
plt.savefig(f"{OUT}/figure1_timeline.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/submission/figure1_timeline.tiff", dpi=300, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 1 (redesigned, problem-first) done")

# ============================================================
# Figure 2: 97 vs 101D -- discrete clustering failure
# ============================================================
d97 = np.load("cache/G2_1B2_results.npz")
d101d = np.load("cache/G2_2_results.npz")

seeds97 = d97["seeds"]
pc1_97 = d97["pc1"]
o500_97 = d97["o500"]
labels97 = d97["kmeans_labels"]

seeds101d = d101d["seeds"]
X101d = d101d["X"]
o500_101d = d101d["o500"]
hier = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X101d)
labels101d = hier.labels_
pc_scores_101d = d101d["pc_scores"]
# Precise failure criterion (matches the established finding exactly):
# 180013 (the high-O500 anchor motivating the search) never becomes its
# own isolated, distinguishing cluster in EITHER paper -- in 97 it shares
# a 19-member majority cluster with 180009; in 101D at k=3 it sits in a
# small 4-member group, never alone. This is the precise, defensible
# claim -- a raw "same cluster as 180009" binary is technically true for
# 97 but MISLEADING for 101D (where 180013 and 180009 land in different
# clusters at k=3, yet neither is meaningfully isolated -- 180013 shares
# its small cluster with 3 other trajectories, not a clean separation).

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

for ax, seeds, pc, o500, labels, title in [
    (axes[0], seeds97, pc1_97, o500_97, labels97, "Paper 97\n(4 features: O100, O1000, tau, H_s)"),
    (axes[1], seeds101d, pc_scores_101d[:, 0], o500_101d, labels101d, "Paper 101D\n(6 non-circular features)"),
]:
    idx13 = list(seeds).index(180013)
    idx09 = list(seeds).index(180009)
    cmap = plt.cm.Set2
    for lab in np.unique(labels):
        mask = labels == lab
        ax.scatter(pc[mask], o500[mask], color=cmap(lab % 8), s=70, alpha=0.75,
                   edgecolor="white", linewidth=0.6, zorder=2)
    ax.scatter(pc[idx13], o500[idx13], marker="*", s=420, color=GOLD13,
               edgecolor="black", linewidth=1.0, zorder=5, label="180013 (high O500)")
    ax.scatter(pc[idx09], o500[idx09], marker="*", s=420, color=GOLD09,
               edgecolor="black", linewidth=1.0, zorder=5, label="180009 (low O500)")
    cluster13_size = int(np.sum(labels == labels[idx13]))
    ax.set_xlabel("Feature-space position (PC1 of feature set)")
    ax.set_ylabel("O500")
    ax.set_title(title + f"\n180013 isolated as own distinguishing cluster: No "
                          f"(shares a {cluster13_size}-member group)", fontsize=9.8)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)

fig.suptitle("Figure 2. Discrete clustering never isolates the high-O500 anchor trajectory,\n"
             "despite strong feature-O500 correlations (not a feature-insufficiency problem)",
             fontsize=12, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig(f"{OUT}/figure2_clustering_failure.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/submission/figure2_clustering_failure.tiff", dpi=300, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 2 done")

# ============================================================
# Figure 3: THE core figure -- now 3 panels (A: failed repr. shows
# signal exists; B: successful repr. separates; C: 8-fold robustness)
# ============================================================
d97 = np.load("cache/G2_1B2_results.npz")
seeds97 = d97["seeds"]
o100_97 = d97["o100"]
o500_97 = d97["o500"]
idx13_97 = list(seeds97).index(180013)
idx09_97 = list(seeds97).index(180009)

d105a = np.load("cache/G2_3_results.npz")
seeds = d105a["seeds"]
pc1 = d105a["pc1"]
o500 = d105a["o500"]
idx13 = list(seeds).index(180013)
idx09 = list(seeds).index(180009)

# exact per-fold leave-one-scale-out values (paper105aext_leaveonescaleout.py)
fold_labels = ["drop B=10", "drop B=20", "drop B=50", "drop B=100",
               "drop B=250", "drop B=1000", "drop B=2000", "drop B=5000"]
fold_r = [0.692, 0.683, 0.682, 0.707, 0.677, 0.687, 0.678, 0.717]

fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.32)

# --- Panel A: failed representation (Paper 97, O100 vs O500), r=0.588 ---
axA = fig.add_subplot(gs[0])
axA.scatter(o100_97, o500_97, s=70, color="#95a5a6", alpha=0.65, edgecolor="white", linewidth=0.6, zorder=2)
axA.scatter(o100_97[idx13_97], o500_97[idx13_97], marker="*", s=420, color=GOLD13,
            edgecolor="black", linewidth=1.0, zorder=5)
axA.scatter(o100_97[idx09_97], o500_97[idx09_97], marker="*", s=420, color=GOLD09,
            edgecolor="black", linewidth=1.0, zorder=5)
rA = np.corrcoef(o100_97, o500_97)[0, 1]
zA = np.polyfit(o100_97, o500_97, 1)
xsA = np.linspace(o100_97.min(), o100_97.max(), 50)
axA.plot(xsA, np.polyval(zA, xsA), "--", color="#333333", alpha=0.5, zorder=1, linewidth=1.2)
axA.set_xlabel("O100 (one feature of the failed\ndiscrete representation, Paper 97)", fontsize=9.5)
axA.set_ylabel("O500 (target variable)", fontsize=10)
axA.set_title(f"A. Signal exists\nr={rA:.3f} -- yet this representation\nfails to separate the golden pair", fontsize=10)
axA.grid(alpha=0.25)

# --- Panel B: successful representation (105A, PC1 vs O500), r=0.540 ---
axB = fig.add_subplot(gs[1])
axB.scatter(pc1, o500, s=70, color="#95a5a6", alpha=0.65, edgecolor="white", linewidth=0.6, zorder=2)
axB.scatter(pc1[idx13], o500[idx13], marker="*", s=420, color=GOLD13, edgecolor="black",
            linewidth=1.0, zorder=5, label="180013 (high O500)")
axB.scatter(pc1[idx09], o500[idx09], marker="*", s=420, color=GOLD09, edgecolor="black",
            linewidth=1.0, zorder=5, label="180009 (low O500)")
rB = np.corrcoef(pc1, o500)[0, 1]
zB = np.polyfit(pc1, o500, 1)
xsB = np.linspace(pc1.min(), pc1.max(), 50)
axB.plot(xsB, np.polyval(zB, xsB), "--", color="#333333", alpha=0.5, zorder=1, linewidth=1.2)
pc_range = pc1.max() - pc1.min()
d_pc = abs(pc1[idx13] - pc1[idx09])
y_bot = o500.min() - 0.85
axB.set_ylim(y_bot - 0.35, o500.max() + 0.3)
axB.annotate("", xy=(pc1[idx13], y_bot), xytext=(pc1[idx09], y_bot),
             arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
axB.text((pc1[idx13] + pc1[idx09]) / 2, y_bot - 0.22,
         f"{100*d_pc/pc_range:.0f}% of PC1's range", ha="center", fontsize=8.5, color="#333333")
axB.set_xlabel("PC1 (continuous compression of the\nmulti-scale O(B) curve)", fontsize=9.5)
axB.set_ylabel("O500 (target variable)", fontsize=10)
axB.set_title(f"B. Not stronger correlation\nr={rB:.3f} -- yet this representation\nseparates the golden pair decisively", fontsize=10)
axB.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
axB.grid(alpha=0.25)

# --- Panel C: 8-fold leave-one-scale-out robustness ---
axC = fig.add_subplot(gs[2])
colors_c = ["#27ae60"] * len(fold_r)
bars = axC.barh(fold_labels, fold_r, color=colors_c, alpha=0.85, edgecolor="white")
axC.axvline(rB, color="#333333", linestyle="--", linewidth=1.2, alpha=0.6, label=f"original (all scales): r={rB:.3f}")
for i, v in enumerate(fold_r):
    axC.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8.5)
axC.set_xlim(0, 0.85)
axC.set_xlabel("PC1-O500 correlation (r)\nafter removing one scale", fontsize=9.5)
axC.set_title("C. Not scale selection\nall 8 folds: r=0.677-0.717,\nALL p<0.001", fontsize=10)
axC.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
axC.grid(alpha=0.25, axis="x")

fig.suptitle("Figure 3. Why the continuous representation succeeds: not signal strength, not correlation\n"
             "magnitude, not a scale-selection artifact", fontsize=13, fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig(f"{OUT}/figure3_pc1_separation.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/submission/figure3_pc1_separation.tiff", dpi=300, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 3 (3-panel) done")

# ============================================================
# Figure 4: representation transition model (schematic)
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.set_xlim(0, 6)
ax.set_ylim(0, 9)
ax.axis("off")

boxes = [
    (6.0, "Wrong representation\n(scalar / ensemble mean)", "#95a5a6"),
    (4.5, "Failed search\n(feature-rich, still discrete)", "#c0392b"),
    (3.0, "New object representation\n(continuous compression)", "#27ae60"),
    (1.5, "Discoverability\n(mechanism becomes tractable)", "#2980b9"),
]
for y, text, color in boxes:
    box = FancyBboxPatch((0.5, y), 5.0, 1.1, boxstyle="round,pad=0.05,rounding_size=0.1",
                          linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.15)
    ax.add_patch(box)
    ax.text(3.0, y + 0.55, text, ha="center", va="center", fontsize=11, fontweight="bold", color=color)

for y_from in [6.0, 4.5, 3.0]:
    ax.annotate("", xy=(3.0, y_from - 0.05), xytext=(3.0, y_from - 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8))

ax.set_title("Figure 4. Representation transition model", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(f"{OUT}/figure4_transition_model.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/submission/figure4_transition_model.tiff", dpi=300, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 4 done")

# ============================================================
# Figure 5: mechanism follows representation (106-108 summary)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
labels = ["Alone: block\n(Mode 1)", "Alone: content\n(Mode 2)", "Predicted\n(additive sum)", "Observed\n(combined)"]
d13_50 = [0.521, 0.151, 0.521 + 0.151, 0.520]
d13_188 = [0.455, 0.134, 0.455 + 0.134, 0.106]
x = np.arange(4)
width = 0.35
ax.bar(x - width / 2, d13_50, width, label="d=50", color="#e67e22", alpha=0.85)
ax.bar(x + width / 2, d13_188, width, label="d=188", color="#c0392b", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Delta accessibility")
ax.set_title("S13: joint block+content interaction\n(repeat-confirmed, Chain 17)", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25, axis="y")
ax.axhline(0, color="black", linewidth=0.8)

ax = axes[1]
hyps = ["2nd-order\nblock grammar", "Source\nimbalance", "Source x block\ninteraction", "Ensemble\nrarity"]
status = [0, 0.3, 0.3, 0]
colors = ["#c0392b", "#e67e22", "#e67e22", "#c0392b"]
ax.barh(hyps, [1, 1, 1, 1], color="#ecf0f1", edgecolor="#bdc3c7")
ax.barh(hyps, status, color=colors, alpha=0.85)
for i, (h, s) in enumerate(zip(hyps, status)):
    label = "EXCLUDED" if s < 0.15 else "inconclusive"
    ax.text(1.02, i, label, va="center", fontsize=9, color=colors[i], fontweight="bold")
ax.set_xlim(0, 1.9)
ax.set_xticks([])
ax.set_title("S19: candidate mechanisms tested\n(exclusion map, Chain 18 + C2-C + C3)", fontsize=10.5)

fig.suptitle("Figure 5. After the representation transition, mechanism becomes tractable\n"
             "but findings remain object-specific, not a general rule",
             fontsize=12, fontweight="bold", y=1.06)
plt.tight_layout()
plt.savefig(f"{OUT}/figure5_mechanism_layer.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/submission/figure5_mechanism_layer.tiff", dpi=300, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 5 done")

print("\nAll figures written to evidence/figures/")
