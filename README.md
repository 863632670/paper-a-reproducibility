# Representation Transitions Reveal Predictive Structure in Complex Systems — Reproduction Repository

Code and data underlying the paper's reported results. Every figure and
statistic in the manuscript traces to one of the scripts and cache
files below.

## Two deliberate deviations from a typical package layout — read before running

1. **`code/` is flat, not split into `simulation/`/`analysis/`/`figures/`
   subfolders.** These scripts import each other directly (e.g.
   `from paper106ab3_5_block_eulerian import block_eulerian_chain`),
   assuming they all sit in the same directory. Splitting them into
   subfolders would break these imports unless every script were
   edited — which risks introducing bugs into an already-verified
   pipeline. Keeping the directory flat preserves exact, tested
   behavior.
2. **The data directory is named `code/cache/`, not a top-level
   `data/raw/` + `data/processed/`.** Every script reads paths like
   `cache/s13_m13.npz` literally, relative to its own working
   directory. Nesting `cache/` directly inside `code/` means every
   script runs correctly with no manual setup, immediately after
   cloning — verified by actually re-running `generate_paperA_figures.py`
   from this exact layout before finalizing the repository, not
   assumed. `cache/` contains both raw trajectory data
   (`s*_traj.npz`, `s*_corrected.npz`, `s13_m13.npz`) and processed/
   derived results (`*_results.npz`) together, as the original
   pipeline produced them.

**To run any script: `cd code/` then run it directly** (e.g.
`python generate_paperA_figures.py`). No path setup, symlinking, or
copying required — `code/cache/` and `code/../figures/` already
resolve correctly from there.

## Structure

```
Paper_A_Repository/
├── README.md                    (this file)
├── LICENSE                      (MIT, draft default -- see note in file)
├── environment/
│   └── requirements.txt         (numpy, scipy, scikit-learn, matplotlib)
├── parameters/
│   └── simulation_config.json   (all simulation parameters, machine-readable)
├── code/                        (33 scripts, flat -- see note above)
│   └── cache/                   (63 files, ~28MB -- raw trajectories + processed results)
└── figures/
    ├── figure1_timeline.png ... figure5_mechanism_layer.png
    └── submission/*.tiff        (300dpi, PLOS-ready)
```

## Reproducing each result

| Manuscript item | Script(s) | Reads from `cache/` |
|---|---|---|
| §2.1: six scalar candidates | `paper94_R100A_reproduction.py`, `paper95_R100A_reproduction.py`, `paper96_R100A_reproduction.py`, `paper98_R100A_reproduction.py`, `paper99_R100A_reproduction.py`, `paper100_R100A_population_upgrade.py` | `s*_corrected.npz` |
| §2.2, attempt 1 (4-feature clustering) | `paper97_G2_1A_ob_only.py`, `paper97_G2_1B2_full_reproduction.py` | `s*_corrected.npz` → produces `G2_1A_results.npz`, `G2_1B2_results.npz` |
| §2.2, attempt 2 (6-feature clustering) | `paper101d_G2_2_reproduction.py` | → `G2_2_results.npz` |
| §2.2, attempt 3 (full structural matrix) | `paper102_R100B_reproduction.py` | |
| §2.2, attempt 4 (surrogate diagnostic) | `paper103b_R100B_reproduction.py` | |
| §2.3 (continuous compression) | `paper104b_R100B_reproduction.py`, `paper105a_G2_3_reproduction.py`, `paper105aext_leaveonescaleout.py` | → `G2_3_results.npz` |
| §2.4, S13 interaction | `paper108_phase1_full_grid.py`, `paper108_phase2a_block_breaking.py`, `paper108_phase2b_grammar_null.py`, `paper108_phase2b_A1_S13_G6_d188_repeats.py`, `paper108_phase2_A2_mode3_mixed.py`, `paper108_phase2_A2_S13_repeat2.py` | `s13_m13.npz`, `s*_traj.npz` |
| §2.4, S19 exclusion map | `paper106_C2A_transition_grammar_mechanism.py`, `paper106_C2B_bias_corrected_entropy.py`, `paper106_C2C_source_organization.py`, `paper108_C3_ensemble_percentile.py` | |
| §2.4, S13 generalization test | `paper106_S13G1_generalization_audit.py` | `s*_corrected.npz` (n=24) |
| All figures | `generate_paperA_figures.py` | reads the `*_results.npz` files above, writes to `figures/` |
| Trajectory population itself | `common106.py` (core simulation), `paper33_size_scaling.py` (checkerboard Metropolis kernel), `gen_pop24_corrected.py`, `gen_s13_m13.py` | generates `cache/s*.npz` from scratch, given only random seeds — the true starting point if reproducing from zero rather than from cached data |

`metrics_tau_hs.py` and `paper100_goldenpair_te.py` are shared utility
modules imported by several scripts above, not run directly.

## Parameters

All simulation parameters (lattice size, temperature, thermalization,
the exact 24-seed trajectory population, the target variable's formal
definition) are in `parameters/simulation_config.json`, matching the
paper's own Supplementary Material Tables S1–S2 exactly.

## Environment

```bash
pip install -r environment/requirements.txt
```

Tested against the package versions listed; no GPU or special hardware
required. Each script completes in seconds to a few minutes on a
standard machine, except the trajectory-generation scripts
(`common106.py`-based simulation), which are the computationally
heaviest step if regenerating `cache/` from scratch rather than using
the provided cached files.

## Citation

If you use this code or data, please cite the paper (full citation to
be added once published) and, if applicable, this repository's own
archived DOI (to be minted upon deposit — see project root for current
status).
