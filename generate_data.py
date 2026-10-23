"""
Step 1: synthetic data. Writes data/train, validation, test (random SNR per voxel) and the paired fixed-SNR
test sets data/snr5 ... snrinf (same voxels and same noise draw at every SNR, only the amplitude differs).
Each set has its own random stream derived from config.SEED.
"""

import os

import numpy as np

from t1t2detr import config as C
from t1t2detr.data import load_protocol, generate_clean, add_noise, save_split


def main():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    protocol = load_protocol()

    for code, (split, n) in enumerate(C.N_SAMPLES.items()):
        rng = np.random.default_rng([C.SEED, code])
        clean, t1, t2, w, n_comp = generate_clean(rng, n, protocol)
        snr = rng.uniform(*C.SNR_RANGE, n)
        save_split(f"{C.DATA_DIR}/{split}.npz", add_noise(clean, snr, rng.standard_normal(clean.shape)),
                   t1, t2, w, n_comp, snr)
        print(f"{split}: {n} voxels, SNR {C.SNR_RANGE[0]:g}-{C.SNR_RANGE[1]:g}")

    rng = np.random.default_rng([C.SEED, len(C.N_SAMPLES)])
    clean, t1, t2, w, n_comp = generate_clean(rng, C.LADDER_SAMPLES, protocol)
    z = rng.standard_normal(clean.shape)
    for snr in C.LADDER_SNR:
        save_split(f"{C.DATA_DIR}/snr{snr:g}.npz", add_noise(clean, snr, z), t1, t2, w, n_comp, snr)
    print(f"SNR ladder: {C.LADDER_SAMPLES} voxels at SNR {', '.join(f'{s:g}' for s in C.LADDER_SNR)}")
    print(f"Saved to {C.DATA_DIR}")


if __name__ == "__main__":
    main()
