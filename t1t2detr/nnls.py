"""
Classical reference: regularized NNLS T1-T2 spectrum (2D inverse Laplace transform), segmented into peaks.

  min ||K f - s||^2 + lam^2 ||f||^2, f >= 0, on a log-spaced (T1, T2) grid with T2 < T1 (zero-order Tikhonov,
  solved as NNLS on [K; lam I]). lam per voxel by the chi2 rule (Whittall & MacKay 1989): the misfit may rise
  to chi2_factor times that of unregularized NNLS. Each grid cell is assigned to the local maximum it climbs
  to; per peak: signal fraction = share of the spectrum, T1/T2 = amplitude-weighted geometric means.
  The signal fraction also serves as the confidence of the peak.
"""

import numpy as np
from scipy.optimize import nnls

from t1t2detr import config as C
from t1t2detr.data import simulate_signal


def build_kernel(protocol, n_grid=C.NNLS["n_grid"]):
    t1, t2 = np.geomspace(*C.T1_RANGE, n_grid), np.geomspace(*C.T2_RANGE, n_grid)
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")
    valid = T2 < T1
    K = simulate_signal(T1[valid][:, None], T2[valid][:, None], np.ones((valid.sum(), 1)), *protocol).T
    return {"K": K, "valid": valid, "t1": t1, "t2": t2}


def _solve(K, s, lam):
    M = K.shape[1]
    A, b = (np.vstack([K, lam * np.eye(M)]), np.r_[s, np.zeros(M)]) if lam > 0 else (K, s)
    f, _ = nnls(A, b, maxiter=50 * M)
    r = K @ f - s
    return f, float(r @ r)


def spectrum(kern, s, factor=C.NNLS["chi2_factor"], lam_range=(1e-5, 10.0), n_iter=12, tol=0.005):
    """Spectrum (n_grid, n_grid) of one voxel signal s by the chi2 rule (bisection in log lam)."""
    s = s / max(np.linalg.norm(s), 1e-30)
    f0, c0 = _solve(kern["K"], s, 0.0)
    target = factor * max(c0, 1e-30)
    lo, hi = np.log(lam_range[0]), np.log(lam_range[1])
    best = f0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f, c = _solve(kern["K"], s, np.exp(mid))
        if c <= target:
            best, lo = f, mid
            if c >= target / (1 + tol):
                break
        else:
            hi = mid
    F = np.zeros(kern["valid"].shape)
    F[kern["valid"]] = best
    return F


def peaks(F, t1, t2, rel_floor=1e-3):
    """(n_peaks, 3) [T1, T2, fraction] by hill climbing in the 8-neighbourhood, sorted by fraction."""
    total = F.sum()
    if total <= 0:
        return np.zeros((0, 3))
    n1, n2 = F.shape
    mask = F > rel_floor * F.max()
    val = np.pad(np.where(mask, F, -1.0), 1, constant_values=-1.0)
    idx = np.pad(np.arange(n1 * n2).reshape(n1, n2), 1, constant_values=-1)
    sh = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    vals = np.stack([val[1 + a:1 + a + n1, 1 + b:1 + b + n2] for a, b in sh])
    ids = np.stack([idx[1 + a:1 + a + n1, 1 + b:1 + b + n2] for a, b in sh])
    up = np.take_along_axis(ids, vals.argmax(0)[None], 0)[0]
    ptr = np.where(mask & (vals.max(0) > F), up, idx[1:-1, 1:-1]).ravel()   # self unless strictly uphill
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    m = mask.ravel()
    _, lab = np.unique(ptr[m], return_inverse=True)
    w = F.ravel()[m]
    L1, L2 = np.meshgrid(np.log(t1), np.log(t2), indexing="ij")
    amp = np.bincount(lab, weights=w)
    out = np.stack([np.exp(np.bincount(lab, weights=w * L1.ravel()[m]) / amp),
                    np.exp(np.bincount(lab, weights=w * L2.ravel()[m]) / amp), amp / total], 1)
    return out[np.argsort(-out[:, 2])]


def predict(kern, s, n_slots=C.NNLS["n_slots"]):
    """(n_slots, 4) [T1, T2, fraction, confidence = fraction]; unused slots far outside the range, conf 0."""
    pk = peaks(spectrum(kern, s), kern["t1"], kern["t2"])[:n_slots]
    out = np.tile([C.T1_RANGE[0] * 1e-3, C.T2_RANGE[0] * 1e-3, 0.0, 0.0], (n_slots, 1))
    out[:len(pk)] = np.c_[pk, pk[:, 2]]
    return out
