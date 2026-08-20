"""Generate and cache the true S13 (source) / M13 (macro) trajectory.
seed=180013, confirmed via Paper 100's golden-pair table and Paper 106D1's title."""
import time
import numpy as np
import common106 as c

t0 = time.time()
source13, macro13 = c.run_trial_combined(seed=180013)
print("elapsed", time.time() - t0, "len", len(source13))
np.savez("cache/s13_m13.npz", source13=source13, macro13=macro13)
print("saved cache/s13_m13.npz")
