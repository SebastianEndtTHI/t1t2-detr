"""
Step 2: trains both DETR variants of config.VARIANTS ("ours" and "diffusion" design) on the same data.
The checkpoint with the highest validation mAP@10 is kept; the learning rate is reduced and training stopped
when it plateaus. Writes models/<variant>/model.pt and history.json.
"""

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS

import numpy as np
import torch

from t1t2detr import config as C
from t1t2detr.data import load_split, normalize_signal, ParamScaler, model_targets
from t1t2detr.loss import HungarianLoss
from t1t2detr.metrics import mean_ap
from t1t2detr.model import T1T2DETR


def build_model(n_inputs):
    return T1T2DETR(n_inputs, **C.MODEL)


@torch.no_grad()
def predict(model, X, scaler, device, batch_size=4096):
    """X: (N, P) raw signals -> (N, Q, 4) [T1 ms, T2 ms, w, existence probability]."""
    model.eval()
    Xn = torch.from_numpy(normalize_signal(X))
    out = np.concatenate([model(Xn[i:i + batch_size].to(device)).cpu().numpy()
                          for i in range(0, len(Xn), batch_size)])
    t1, t2 = scaler.denorm(out[..., 0].astype(float), out[..., 1].astype(float))
    return np.stack([t1, t2, out[..., 2], 1 / (1 + np.exp(-out[..., 3]))], -1)


def train_variant(name, device):
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    v = C.VARIANTS[name]
    scaler = ParamScaler(v["param_scale"])
    tr, va = load_split("train"), load_split("validation")
    X = torch.from_numpy(normalize_signal(tr["signal"])).to(device)
    Y = torch.from_numpy(model_targets(tr, scaler)).to(device)
    NC = torch.from_numpy(tr["n_comp"]).to(device)

    model = build_model(X.shape[1]).to(device)
    crit = HungarianLoss(v["weighting"], C.LOSS_WEIGHTS)
    T = C.TRAIN
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"], betas=T["betas"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=T["lr_factor"],
                                                       patience=T["lr_patience"], min_lr=T["min_lr"])
    out_dir = os.path.join(C.MODEL_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(C.SEED)
    hist, best, wait = {"train_loss": [], "val_mAP": [], "lr": []}, -1.0, 0
    print(f"\n=== {name}: {v} | {sum(p.numel() for p in model.parameters()):,} parameters | {device}")

    for epoch in range(T["max_epochs"]):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(X), generator=gen).to(device)
        losses = []
        for i in range(0, len(X), T["batch_size"]):
            b = perm[i:i + T["batch_size"]]
            loss = crit(model(X[b]), Y[b], NC[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"])
            opt.step()
            losses.append(loss.item())

        val_map = mean_ap(predict(model, va["signal"], scaler, device), va, C.TAU)
        sched.step(val_map)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_mAP"].append(val_map)
        hist["lr"].append(opt.param_groups[0]["lr"])
        if val_map > best:
            best, wait, hist["best_epoch"] = val_map, 0, epoch + 1
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        else:
            wait += 1
        print(f"epoch {epoch + 1:3d}  train loss {hist['train_loss'][-1]:.5f}  val mAP@{round(C.TAU * 100)} "
              f"{val_map:.4f}  lr {hist['lr'][-1]:.1e}  {time.time() - t0:.0f} s", flush=True)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        if wait >= T["patience"]:
            print(f"early stopping, best epoch {hist['best_epoch']} (val mAP {best:.4f})")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(C.VARIANTS), choices=list(C.VARIANTS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    for name in args.variants:
        train_variant(name, args.device)


if __name__ == "__main__":
    main()
