"""
Step 3: NNLS reference on the validation, test and SNR-ladder sets (CPU, parallel over voxels).
Writes predictions/nnls_<case>.npz in the DETR prediction format.
"""

import argparse
import os
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                              # one BLAS thread per worker, overriding cluster defaults

import numpy as np
import scipy

from t1t2detr import config as C
from t1t2detr.data import load_protocol, load_split, ladder_cases
from t1t2detr.nnls import build_kernel, predict

_KERNEL = {}


def _init():
    _KERNEL.update(build_kernel(load_protocol()))


def _work(s):
    t = time.process_time()                           # CPU time of this worker: unaffected by oversubscription
    return predict(_KERNEL, s.astype(float)), time.process_time() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", default=int(os.environ.get("SLURM_CPUS_PER_TASK", 40)), type=int)
    p.add_argument("--n_timing", default=200, type=int, help="test voxels timed in a single process")
    args = p.parse_args()
    if tuple(int(v) for v in scipy.__version__.split(".")[:2]) < (1, 16):
        print(f"WARNING: SciPy {scipy.__version__}: nnls is much slower before 1.16 (pip install 'scipy>=1.16')",
              flush=True)
    os.makedirs(C.PRED_DIR, exist_ok=True)
    with Pool(args.workers, initializer=_init) as pool:
        for case in ["validation", "test"] + ladder_cases():
            S = load_split(case)["signal"]
            t = time.time()
            res = pool.map(_work, S, chunksize=max(1, len(S) // (8 * args.workers)))
            sec = np.array([r[1] for r in res])
            np.savez(f"{C.PRED_DIR}/nnls_{case}.npz", pred=np.stack([r[0] for r in res]),
                     sec_per_voxel=sec.mean(), device="1 CPU core")
            print(f"{case}: {len(S)} voxels, {sec.mean() * 1e3:.0f} ms/voxel per core, "
                  f"{time.time() - t:.0f} s wall", flush=True)

    # Runtime for Table 1: one process alone (parallel workers share cores and L3 cache, which inflates their times)
    _init()
    S = load_split("test")["signal"][:args.n_timing]
    t = time.perf_counter()
    for s in S:
        _work(s)
    sec = (time.perf_counter() - t) / len(S)
    with np.load(f"{C.PRED_DIR}/nnls_test.npz") as f:
        pred = f["pred"]
    np.savez(f"{C.PRED_DIR}/nnls_test.npz", pred=pred, sec_per_voxel=sec, device="1 CPU core, single process")
    print(f"runtime (single process, {len(S)} test voxels): {sec * 1e3:.0f} ms/voxel", flush=True)


if __name__ == "__main__":
    main()
