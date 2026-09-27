import numpy as np
import scipy.sparse as sp
import time

print("Creating fake data...")
data = np.ones(300000000, dtype=np.float32)
indices = np.random.randint(0, 30000, size=300000000, dtype=np.int32)
indptr = np.arange(0, 300000001, dtype=np.int32)

stop_cols = np.arange(500)

t0 = time.time()
print("Running np.isin...")
mask = np.isin(indices, stop_cols)
data[mask] = 0.0
print(f"Done in {time.time()-t0:.2f}s")
