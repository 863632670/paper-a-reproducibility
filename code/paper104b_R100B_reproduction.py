"""
R100-B: Paper 104B reproduction ("Two Qualitatively Different
Sensitivity Shapes, Not Just Magnitudes"). Computes O(B) across the
full block-size grid B in {10,20,50,100,250,500,1000,5000} for the
golden pair (180013, 180009).

Paper 104B's own reported values:
  180013: B=10:-0.404  B=20:-0.059  B=50:0.465  B=100:1.061  B=250:1.404
          B=500:1.993  B=1000:0.801  B=5000:0.543
  180009: deep negative -2.4 to -3.4 across B=10-500 (min ~B=100),
          -1.704 at B=1000, +0.888 at B=5000
"""
import numpy as np
import common106 as c

B_grid = [10, 20, 50, 100, 250, 500, 1000, 5000]
paper_13 = {10: -0.404, 20: -0.059, 50: 0.465, 100: 1.061, 250: 1.404,
            500: 1.993, 1000: 0.801, 5000: 0.543}

for seed in [180013, 180009]:
    suffix = seed - 180000
    d = np.load(f"cache/s{suffix:02d}_corrected.npz")
    source, macro = d["source"], d["macro"]
    print(f"\n=== seed {seed} ===")
    for B in B_grid:
        ob = c.O_of_B(source, macro, B, rng_seed=0)
        if seed == 180013:
            print(f"B={B:5d}: O(B)={ob:7.3f}   (paper: {paper_13[B]:.3f})")
        else:
            print(f"B={B:5d}: O(B)={ob:7.3f}")

print("\n=== 180009 shape check vs paper's qualitative description ===")
print("paper: deep negative (-2.4 to -3.4) across B=10-500, min near B=100,")
print("       rising to -1.704 at B=1000, reversing to +0.888 at B=5000")
