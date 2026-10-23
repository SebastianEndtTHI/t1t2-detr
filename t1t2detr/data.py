"""Forward model, synthetic voxels, data files and normalization."""

import numpy as np
import scipy.io as sio

from t1t2detr import config as C

K_MAX = max(C.N_COMP)


def load_protocol(path=C.PROTOCOL_FILE):
    """TI/TE pairs in scanner order plus TR, all in ms."""
    d = sio.loadmat(path)
    return (np.asarray(d["ti"], float).ravel(), np.asarray(d["te"], float).ravel(),
            float(np.asarray(d["tr"]).ravel()[0]))


def simulate_signal(t1, t2, w, ti, te, tr):
    """
    Noise-free inversion-recovery multi-echo signal of N voxels with K compartments:
        S_p = sum_k w_k (1 - 2 exp(-TI_p/T1_k) + exp(-TR/T1_k)) exp(-TE_p/T2_k)
    t1, t2, w: (N, K) in ms; padded compartments need w = 0 and finite T1/T2. Returns (N, P).
    """
    t1, t2 = t1[:, None, :], t2[:, None, :]
    k1 = 1.0 - 2.0 * np.exp(-ti[None, :, None] / t1) + np.exp(-tr / t1)
    return np.einsum("npk,nk->np", k1 * np.exp(-te[None, :, None] / t2), w)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic voxels
# ─────────────────────────────────────────────────────────────────────────
def sample_t1_t2(rng, n):
    """Log-uniform T1 and T2, redrawn until T2 < T1."""
    t1, t2 = np.empty(n), np.empty(n)
    todo = np.arange(n)
    while todo.size:
        a = np.exp(rng.uniform(*np.log(C.T1_RANGE), todo.size))
        b = np.exp(rng.uniform(*np.log(C.T2_RANGE), todo.size))
        ok = b < a
        t1[todo[ok]], t2[todo[ok]] = a[ok], b[ok]
        todo = todo[~ok]
    return t1, t2


def generate_clean(rng, n, protocol):
    """n voxels, compartment counts equally represented. Returns clean signals (n, P), t1, t2, w (n, K_MAX)
    NaN-padded, and n_comp."""
    n_comp = np.concatenate([np.repeat(C.N_COMP, n // len(C.N_COMP)),
                             rng.choice(C.N_COMP, n % len(C.N_COMP))])
    rng.shuffle(n_comp)
    exists = np.arange(K_MAX)[None, :] < n_comp[:, None]

    t1, t2 = (a.reshape(n, K_MAX) for a in sample_t1_t2(rng, n * K_MAX))
    w = np.zeros((n, K_MAX))
    for nc in C.N_COMP:                                   # Dirichlet on the part above the minimum weight
        idx = np.where(n_comp == nc)[0]
        w[idx, :nc] = rng.dirichlet(np.ones(nc), size=idx.size) * (1 - nc * C.MIN_WEIGHT) + C.MIN_WEIGHT
    clean = simulate_signal(t1, t2, w, *protocol)
    t1[~exists], t2[~exists], w[~exists] = np.nan, np.nan, np.nan
    return clean, t1, t2, w, n_comp


def add_noise(clean, snr, z):
    """Gaussian noise (the IR signal is signed) with sigma = max|S_clean| / SNR; z standard normal."""
    return clean + (np.abs(clean).max(axis=1, keepdims=True) / np.reshape(snr, (-1, 1))) * z


def save_split(path, signal, t1, t2, w, n_comp, snr):
    np.savez(path, signal=signal.astype(np.float32), t1=t1, t2=t2, w=w, n_comp=n_comp,
             snr=np.broadcast_to(np.asarray(snr, float), len(signal)))


def load_split(case, data_dir=C.DATA_DIR):
    """case: 'train', 'validation', 'test' or a ladder rung 'snr5' ... 'snrinf'."""
    with np.load(f"{data_dir}/{case}.npz") as f:
        return {k: f[k] for k in f.files}


def ladder_cases():
    return [f"snr{s:g}" for s in C.LADDER_SNR]


# ─────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────
def normalize_signal(X):
    """Unit L2 norm per voxel (removes M0 and receiver scaling)."""
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return (X / np.where(n == 0, 1.0, n)).astype(np.float32)


class ParamScaler:
    """T1/T2 [ms] <-> [0, 1]. 'log': log-min-max over the ranges; 'linear': divided by the range maximum
    (as MD / 4 in the diffusion DETR)."""

    def __init__(self, scale):
        if scale not in ("log", "linear"):
            raise ValueError(scale)
        self.scale = scale

    def _fwd(self, x, rng):
        if self.scale == "log":
            return np.clip((np.log(x) - np.log(rng[0])) / np.log(rng[1] / rng[0]), 0, 1)
        return np.clip(x / rng[1], 0, 1)

    def _inv(self, y, rng):
        if self.scale == "log":
            return np.exp(y * np.log(rng[1] / rng[0]) + np.log(rng[0]))
        return y * rng[1]

    def norm(self, t1, t2):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._fwd(t1, C.T1_RANGE), self._fwd(t2, C.T2_RANGE)

    def denorm(self, y1, y2):
        return self._inv(y1, C.T1_RANGE), self._inv(y2, C.T2_RANGE)


def model_targets(d, scaler):
    """(N, K_MAX * 3) [T1, T2, w] normalized, zero-padded."""
    t1n, t2n = scaler.norm(d["t1"], d["t2"])
    y = np.nan_to_num(np.stack([t1n, t2n, d["w"]], axis=2), nan=0.0)
    return y.reshape(len(y), -1).astype(np.float32)
