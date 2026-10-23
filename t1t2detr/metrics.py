"""
Detection metrics. Predictions are arrays (N, Q, 4) [T1 ms, T2 ms, w, confidence]; ground truth is a data dict
(load_split). Normalized distance (ND): a prediction matches a true compartment if
|log T1_pred - log T1_true| / log-span(T1) <= tau and the same for T2.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from t1t2detr import config as C

SPANS = np.log([C.T1_RANGE[1] / C.T1_RANGE[0], C.T2_RANGE[1] / C.T2_RANGE[0]])


def nd_records(pred, d, tau):
    """Assigns every query to the closest true compartment within tau (smallest ND sum), or to none.
    Returns flat per-query records: sample, conf, gt (-1 = no match), gid (global id of the true compartment)."""
    N, Q, _ = pred.shape
    G = np.stack([d["t1"], d["t2"]], -1)                                        # (N, K, 2)
    K = G.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = np.abs(np.log(pred[:, :, None, :2]) - np.log(G[:, None])) / SPANS  # (N, Q, K, 2)
    ok = (nd <= tau).all(-1) & (np.arange(K)[None, None, :] < d["n_comp"][:, None, None])
    score = np.where(ok, np.nan_to_num(nd.sum(-1), nan=np.inf), np.inf)
    best = score.argmin(-1)
    gt = np.where(np.isfinite(np.take_along_axis(score, best[..., None], -1)[..., 0]), best, -1).ravel()
    sample = np.repeat(np.arange(N), Q)
    return {"sample": sample, "conf": pred[..., 3].ravel(), "gt": gt, "gid": np.where(gt >= 0, sample * K + gt, -1),
            "Q": Q, "K": K}


def average_precision(conf, gid, n_gt):
    """101-point interpolated AP (COCO). Repeated matches of one true compartment count as false positives."""
    if n_gt == 0 or len(conf) == 0:
        return np.nan if n_gt == 0 else 0.0
    order = np.argsort(-conf, kind="stable")
    gid = gid[order]
    tp = np.zeros(len(gid), bool)
    m = np.where(gid >= 0)[0]
    tp[m[np.unique(gid[m], return_index=True)[1]]] = True
    acc = np.cumsum(tp)
    recall, precision = acc / n_gt, acc / np.arange(1, len(gid) + 1)
    env = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(env), env[np.minimum(idx, len(env) - 1)], 0.0).mean())


def resampled_map(rec, n_comp, idx):
    """mAP on the voxels idx (repetitions allowed: every drawn copy gets its own ids)."""
    Q, K = rec["Q"], rec["K"]
    r = (idx[:, None] * Q + np.arange(Q)).ravel()
    gt = rec["gt"][r]
    gid = np.where(gt >= 0, np.repeat(np.arange(len(idx)), Q) * K + gt, -1)
    return average_precision(rec["conf"][r], gid, int(n_comp[idx].sum()))


def mean_ap(pred, d, tau):
    rec = nd_records(pred, d, tau)
    return average_precision(rec["conf"], rec["gid"], int(d["n_comp"].sum()))


def voxel_counts(rec, n_voxels, thr):
    """Kept predictions and true positives per voxel (each true compartment counted once)."""
    keep = rec["conf"] >= thr
    m = keep & (rec["gid"] >= 0)
    order = np.where(m)[0][np.argsort(-rec["conf"][m], kind="stable")]
    tp = order[np.unique(rec["gid"][order], return_index=True)[1]]
    return (np.bincount(rec["sample"][keep], minlength=n_voxels),
            np.bincount(rec["sample"][tp], minlength=n_voxels))


def count_metrics(n_pred, n_tp, n_comp):
    """F1, strict accuracy (right count and every compartment matched) and count accuracy."""
    return {"F1": 2 * n_tp.sum() / max(n_pred.sum() + n_comp.sum(), 1),
            "strict_acc": float(np.mean((n_pred == n_comp) & (n_tp == n_comp))),
            "count_acc": float(np.mean(n_pred == n_comp))}


def best_threshold(pred, d, tau=C.TAU, grid=np.round(np.arange(0.0, 1.0, 0.01), 2)):
    """Confidence threshold with the highest F1 (used on validation data)."""
    rec = nd_records(pred, d, tau)
    f1 = [count_metrics(*voxel_counts(rec, len(d["n_comp"]), t), d["n_comp"])["F1"] for t in grid]
    return float(grid[int(np.argmax(f1))])


def grouped_map(pred, d, tau=C.TAU):
    """mAP per compartment count and per signal-fraction group (n_c = 1 is its own group).
    Matched predictions are grouped by their true compartment, unmatched ones by the voxel's n_c or their
    own predicted weight."""
    rec = nd_records(pred, d, tau)
    K, Q = rec["K"], rec["Q"]
    nc_vox = d["n_comp"]
    valid = (np.arange(K)[None, :] < nc_vox[:, None]).ravel()
    gidx = np.maximum(rec["gid"], 0)

    def bucket_map(gt_bucket, rec_bucket, labels):
        out = {}
        for b, lab in enumerate(labels):
            n_gt = int(np.sum(gt_bucket == b))
            m = rec_bucket == b
            out[lab] = average_precision(rec["conf"][m], rec["gid"][m], n_gt) if n_gt else np.nan
        return out

    nc_gt = np.where(valid, np.repeat(nc_vox, K), -1)
    nc_rec = nc_vox[rec["sample"]]
    by_nc = bucket_map(nc_gt, nc_rec, [str(n) if n in C.N_COMP else "" for n in range(max(C.N_COMP) + 1)])
    by_nc = {k: v for k, v in by_nc.items() if k}

    e = C.WEIGHT_EDGES
    n_w = len(e) - 1
    to_b = lambda w: np.clip(np.digitize(w, e) - 1, 0, n_w - 1)
    w_gt = np.where(valid, np.where(np.repeat(nc_vox, K) == 1, n_w, to_b(np.nan_to_num(d["w"].ravel()))), -1)
    w_unm = np.where(nc_rec == 1, n_w, to_b(pred[..., 2].ravel()))
    w_rec = np.where(rec["gid"] >= 0, w_gt[gidx], w_unm)
    labels = [f"{e[i]:.2f}<=w<{e[i + 1]:.2f}" for i in range(n_w)] + ["n_c=1"]
    return by_nc, bucket_map(w_gt, w_rec, labels)


def matched_pairs(pred, d, thr):
    """Hungarian matching (log T1, log T2, w) between kept predictions and true compartments, for scatter
    plots. Returns true and predicted (n, 3) arrays [T1, T2, w]."""
    true, est = [], []
    for i in range(len(pred)):
        p = pred[i][pred[i, :, 3] >= thr]
        k = d["n_comp"][i]
        if len(p) == 0:
            continue
        g = np.stack([d["t1"][i, :k], d["t2"][i, :k], d["w"][i, :k]], 1)
        cost = (((np.log(p[:, None, :2]) - np.log(g[None, :, :2])) / SPANS) ** 2).sum(-1) \
            + (p[:, None, 2] - g[None, :, 2]) ** 2
        r, c = linear_sum_assignment(cost)
        true.append(g[c]); est.append(p[r, :3])
    return np.concatenate(true), np.concatenate(est)


def r2(true, pred):
    return float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))


# ─────────────────────────────────────────────────────────────────────────
# Paired bootstrap
# ─────────────────────────────────────────────────────────────────────────
def bootstrap(preds, d, thresholds, taus, n_boot, rng):
    """
    preds: {method: (N, Q, 4)} on the same voxels. The same resamples of voxel indices are used for all
    methods (paired), so differences between methods exclude the variation in voxel difficulty.
    Returns {method: {metric: (point, (B,) bootstrap values)}}.
    """
    N = len(d["n_comp"])
    idx = rng.integers(0, N, size=(n_boot, N))
    out = {}
    for m, p in preds.items():
        res = {}
        for tau in taus:
            rec = nd_records(p, d, tau)
            res[f"mAP@{round(tau * 100)}"] = (resampled_map(rec, d["n_comp"], np.arange(N)),
                                             np.array([resampled_map(rec, d["n_comp"], i) for i in idx]))
        n_pred, n_tp = voxel_counts(nd_records(p, d, C.TAU), N, thresholds[m])
        point = count_metrics(n_pred, n_tp, d["n_comp"])
        boot = [count_metrics(n_pred[i], n_tp[i], d["n_comp"][i]) for i in idx]
        for k in point:
            res[k] = (point[k], np.array([b[k] for b in boot]))
        out[m] = res
    return out


def ci(x, level=C.CI_LEVEL):
    a = (1 - level) / 2
    return [float(np.quantile(x, a)), float(np.quantile(x, 1 - a))]
