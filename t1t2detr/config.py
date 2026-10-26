"""
All settings of the paper experiments. Every script takes its defaults from here, so running the four
scripts without arguments reproduces the reported numbers and figures.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROTOCOL_FILE = os.path.join(REPO_ROOT, "protocols", "ti_te_dict.mat")
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
PRED_DIR = os.path.join(REPO_ROOT, "predictions")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# one seed for everything (data, training, bootstrap, example selection)
SEED = 42

# ── synthetic data ──────────────────────────────────────────────────────
T1_RANGE = (50.0, 3500.0)                # ms, log-uniform, with T2 < T1
T2_RANGE = (5.0, 500.0)                  # ms
N_COMP = (1, 2, 3, 4)                    # compartments per voxel, equally represented
MIN_WEIGHT = 0.05                        # minimum signal fraction
SNR_RANGE = (10.0, 150.0)                # per-voxel SNR = max|S_clean| / sigma, uniform
N_SAMPLES = {"train": 200_000, "validation": 25_000, "test": 25_000}
LADDER_SNR = (5, 10, 20, 40, 80, 120, 200, float("inf"))   # paired fixed-SNR test sets
LADDER_SAMPLES = 5_000

# ── model and training ──────────────────────────────────────────────────
# "ours": log-scaled T1/T2 targets, uniform loss weighting.
# "diffusion": the design of the diffusion DETR (targets scaled linearly by their maximum, like MD/4,
# and T1/T2 errors weighted by the true signal fraction); everything else identical.
VARIANTS = {"ours": {"param_scale": "log", "weighting": "uniform"},
            "diffusion": {"param_scale": "linear", "weighting": "signal_fraction"}}
MODEL = {"hidden_dim": 512, "fs_dim": 256, "n_queries": 10, "n_layers": 4, "n_heads": 4}
LOSS_WEIGHTS = {"t1": 1.0, "t2": 1.0, "w": 1.0, "exist": 0.1}
TRAIN = {"lr": 1e-4, "weight_decay": 1e-4, "betas": (0.9, 0.98), "batch_size": 512, "max_epochs": 500,
         "patience": 35, "lr_factor": 0.5, "lr_patience": 7, "min_lr": 1e-6, "grad_clip": 1.0}

# ── NNLS reference ──────────────────────────────────────────────────────
NNLS = {"n_grid": 30, "chi2_factor": 1.02, "n_slots": 10}

# ── evaluation ──────────────────────────────────────────────────────────
TAU = 0.10                               # main ND tolerance (threshold search, F1, accuracies, groups)
TAUS = (0.05, 0.10)                      # reported mAP tolerances
N_BOOT = 1000
CI_LEVEL = 0.95
WEIGHT_EDGES = (0.05, 0.35, 0.65, 1.0)   # signal-fraction groups for n_c >= 2
EXAMPLE_SNR = 40                         # ladder rung of the example spectra (Fig. 1)
EXAMPLE_NCOMP = (2, 3, 4)
SPECTRA_AXES = "log"                     # "log" or "linear" for Fig. 1
METHODS = {"nnls": "NNLS", "diffusion": "DETR (diff)", "ours": "DETR (ours)"}
COLORS = {"nnls": "#E37222", "diffusion": "#A2AD00", "ours": "#0065BD"}
