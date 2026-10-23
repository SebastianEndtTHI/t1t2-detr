"""
Step 4: all results of the paper. Runs the DETR models on the validation, test and SNR-ladder sets, chooses
every method's confidence threshold on validation (best F1), and writes to results/:
  table1.md / table1.tex   NNLS vs. DETR (diffusion design) vs. DETR (ours) on the test set
  fig1_spectra.pdf         example voxels: NNLS spectrum and peaks, DETR (ours), ground truth
  fig2_snr.pdf             mAP and strict accuracy vs. SNR (paired ladder) with bootstrap CIs
  fig3_scatter.pdf         true vs. predicted T1, T2, weight of DETR (ours)
  results.json             all numbers, including confidence intervals
"""

import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from t1t2detr import config as C
from t1t2detr import metrics as M
from t1t2detr.data import load_protocol, load_split, ladder_cases, ParamScaler
from t1t2detr.nnls import build_kernel, spectrum, peaks
from t1t2detr.plots import fig_spectra, fig_snr, fig_scatter
from train import build_model, predict

CASES = ["validation", "test"] + ladder_cases()


def run_detr(device):
    """DETR predictions for all cases; returns {variant: test-set seconds per voxel}."""
    runtime = {}
    for name, v in C.VARIANTS.items():
        scaler = ParamScaler(v["param_scale"])
        for case in CASES:
            S = load_split(case)["signal"]
            model = build_model(S.shape[1]).to(device)
            model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, name, "model.pt"), map_location=device,
                                             weights_only=True))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            P = predict(model, S, scaler, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if case == "test":
                runtime[name] = (time.perf_counter() - t) / len(S)
            np.savez(f"{C.PRED_DIR}/{name}_{case}.npz", pred=P)
        print(f"DETR ({name}): predictions written")
    return runtime


def load_pred(method, case):
    path = f"{C.PRED_DIR}/{method}_{case}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing" + (" - run run_nnls.py first" if method == "nnls" else ""))
    with np.load(path) as f:
        return f["pred"].astype(float)


def boot_summary(boot, ref="ours"):
    """Point values with CIs, and differences ours - other with paired CIs."""
    out = {m: {k: [p, M.ci(b)] for k, (p, b) in res.items()} for m, res in boot.items()}
    for m in boot:
        if m != ref:
            out[f"{ref} - {m}"] = {k: [boot[ref][k][0] - p, M.ci(boot[ref][k][1] - b)] for k, (p, b) in boot[m].items()}
    return out


def _ms(sec):
    ms = sec * 1e3
    return f"{ms:.0f}" if ms >= 10 else f"{ms:.2g}"


def write_table(res, grouped, runtime, device, path):
    t5, t10 = (f"mAP@{round(t * 100)}" for t in C.TAUS)
    wl = list(next(iter(grouped.values()))[1])[:-1]                     # weight groups without "n_c=1"
    w_rng = [w.split("<=w<") for w in wl]
    head_md = ["Method", t5, t10, "strict acc. [%]"] + [f"n_c={n}" for n in C.N_COMP] \
        + [f"w {lo}–{hi}" for lo, hi in w_rng] + ["ms/voxel"]
    vals = {m: [res[m][t5][0], res[m][t10][0], 100 * res[m]["strict_acc"][0]]
            + [grouped[m][0][str(n)] for n in C.N_COMP] + [grouped[m][1][w] for w in wl] for m in C.METHODS}
    fmt = lambda j, v: f"{v:.1f}" if j == 2 else f"{v:.3f}"
    best = [max(vals[m][j] for m in C.METHODS) for j in range(len(head_md) - 2)]
    rows_md = []
    for m in C.METHODS:
        cells = [fmt(j, v) for j, v in enumerate(vals[m])]
        rows_md.append([C.METHODS[m]] + cells + [_ms(runtime[m])])
    notes = []
    for m in C.METHODS:
        if m == "ours":
            continue
        d = res[f"ours - {m}"]
        notes.append(f"{C.METHODS['ours']} − {C.METHODS[m]}: " + ", ".join(
            f"Δ{k.replace('_', ' ')} {d[k][0]:+.3f} [{d[k][1][0]:+.3f}, {d[k][1][1]:+.3f}]" for k in (t5, t10, "strict_acc")))
    foot = (f"mAP per n_c and signal-fraction group at tau = {C.TAU:g}. Runtime: NNLS {runtime['device_nnls']}, "
            f"DETR {device}. Differences with paired bootstrap {round(100 * C.CI_LEVEL)} % CIs "
            f"({C.N_BOOT} resamples over voxels):")
    md = ["| " + " | ".join(head_md) + " |", "|" + "---|" * len(head_md)] + ["| " + " | ".join(r) + " |" for r in rows_md]
    with open(path + ".md", "w") as f:
        f.write("\n".join(md + ["", foot] + [f"- {n}" for n in notes]) + "\n")
    print("\n".join(md + ["", foot] + notes))
    print(f"Saved: {path}.md")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(C.PRED_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    runtime = run_detr(device)
    with np.load(f"{C.PRED_DIR}/nnls_test.npz") as f:
        runtime["nnls"], runtime["device_nnls"] = float(f["sec_per_voxel"]), str(f["device"])
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    data = {c: load_split(c) for c in CASES}
    preds = {c: {m: load_pred(m, c) for m in C.METHODS} for c in CASES}
    thr = {m: M.best_threshold(preds["validation"][m], data["validation"]) for m in C.METHODS}
    print("confidence thresholds (validation, best F1):", {m: round(t, 2) for m, t in thr.items()})
    rng = np.random.default_rng([C.SEED, 100])

    # Table 1: test set
    res = boot_summary(M.bootstrap(preds["test"], data["test"], thr, C.TAUS, C.N_BOOT, rng))
    grouped = {m: M.grouped_map(preds["test"][m], data["test"]) for m in C.METHODS}
    write_table(res, grouped, runtime, dev_name, os.path.join(C.RESULTS_DIR, "table1"))

    # Fig. 2: SNR ladder
    ladder = {m: {"snr": list(C.LADDER_SNR), f"mAP@{round(C.TAU * 100)}": [], "strict_acc": []} for m in C.METHODS}
    for case in ladder_cases():
        r = boot_summary(M.bootstrap(preds[case], data[case], thr, (C.TAU,), C.N_BOOT, rng))
        for m in C.METHODS:
            for k in (f"mAP@{round(C.TAU * 100)}", "strict_acc"):
                ladder[m][k].append(r[m][k])
        print(f"{case}: " + "  ".join(f"{C.METHODS[m]} {r[m][f'mAP@{round(C.TAU * 100)}'][0]:.3f}" for m in C.METHODS))
    fig_snr(ladder, os.path.join(C.RESULTS_DIR, "fig2_snr.pdf"))

    # Fig. 1: example spectra
    case = f"snr{C.EXAMPLE_SNR:g}"
    d, kern = data[case], build_kernel(load_protocol())
    ex_rng = np.random.default_rng([C.SEED, 200])
    examples = []
    for nc in C.EXAMPLE_NCOMP:
        i = int(ex_rng.choice(np.where(d["n_comp"] == nc)[0]))
        F = spectrum(kern, d["signal"][i].astype(float))
        pk = peaks(F, kern["t1"], kern["t2"])
        p = preds[case]["ours"][i]
        examples.append({"n_comp": nc, "snr": C.EXAMPLE_SNR, "index": i, "spectrum": F,
                         "gt": np.stack([d["t1"][i, :nc], d["t2"][i, :nc], d["w"][i, :nc]], 1),
                         "nnls": pk[pk[:, 2] >= thr["nnls"]], "detr": p[p[:, 3] >= thr["ours"], :3]})
    fig_spectra(examples, kern, os.path.join(C.RESULTS_DIR, "fig1_spectra.pdf"))

    # Fig. 3: scatter of DETR (ours) on the test set
    true, est = M.matched_pairs(preds["test"]["ours"], data["test"], thr["ours"])
    fig_scatter(true, est, os.path.join(C.RESULTS_DIR, "fig3_scatter.pdf"))

    out = {"seed": C.SEED, "thresholds": thr, "test": res, "test_grouped_mAP": {m: {"by_n_comp": g[0], "by_weight": g[1]}
           for m, g in grouped.items()}, "runtime_s_per_voxel": {m: runtime[m] for m in C.METHODS},
           "ladder": ladder, "examples": [{"n_comp": e["n_comp"], "index": e["index"]} for e in examples]}
    with open(os.path.join(C.RESULTS_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {os.path.join(C.RESULTS_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
