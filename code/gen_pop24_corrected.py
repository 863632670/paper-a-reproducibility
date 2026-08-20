"""
G2-0: Generate the CORRECTED 24-trajectory population, seeds 180000
through 180023 (the literal, sequential range confirmed via direct
transcript evidence in PCA_Prototype_Lineage.md -- Paper 97's own
seed-by-seed O(100)/O(1000) printout and Paper 101D's clustering output
matching seed 180009's O500=-2.065 exactly). This SUPERSEDES
gen_pop24.py's population (10 reused + 14 independently-drawn via
RandomState seed=42), which was built under the now-corrected
assumption that the true population was unrecoverable.

Reuses already-cached trajectories where the suffix falls in 00-23
(9 of them: 02,03,07,09,10,15,18,19,21, plus 13 under s13_m13.npz's
own key names) and generates the 15 missing ones (00,01,04,05,06,08,
11,12,13,14,16,17,20,22,23 -- note 13 is regenerated fresh here under
the standard s13_traj.npz naming for consistency with the other 23,
even though s13_m13.npz already has it under different key names).
"""
import numpy as np
import common106 as c

all_seeds = list(range(180000, 180024))
already_cached_suffixes = {2, 3, 7, 9, 10, 15, 18, 19, 21}

for seed in all_seeds:
    suffix = seed - 180000
    fname = f"cache/s{suffix:02d}_corrected.npz"
    if suffix in already_cached_suffixes:
        d = np.load(f"cache/s{suffix:02d}_traj.npz")
        source, macro = d["source"], d["macro"]
        print(f"seed {seed}: reused from existing cache/s{suffix:02d}_traj.npz")
    elif suffix == 13:
        d = np.load("cache/s13_m13.npz")
        source, macro = d["source13"], d["macro13"]
        print(f"seed {seed}: reused from existing cache/s13_m13.npz")
    else:
        source, macro = c.run_trial_combined(seed)
        print(f"seed {seed}: GENERATED fresh via run_trial_combined")
    np.savez(fname, source=source, macro=macro, seed=seed)

np.save("cache/pop24_corrected_seeds.npy", np.array(all_seeds))
print("\nDone. All 24 trajectories (seeds 180000-180023) cached under cache/s{NN}_corrected.npz")
