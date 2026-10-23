"""Hungarian set loss."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianLoss(torch.nn.Module):
    """
    y_pred: (B, Q, 4) [T1, T2, w, existence logit], y_true: (B, K * 3) [T1, T2, w], n_comp: (B,).
    weighting 'uniform': every compartment counts the same; 'signal_fraction': T1/T2 errors scaled by the
    true signal fraction (as in the diffusion DETR). The scaling enters the matching cost as well.
    """

    def __init__(self, weighting, lw):
        super().__init__()
        if weighting not in ("uniform", "signal_fraction"):
            raise ValueError(weighting)
        self.weighting, self.lw = weighting, lw

    def forward(self, y_pred, y_true, n_comp):
        B, Q, _ = y_pred.shape
        y_true = y_true.reshape(B, -1, 3)
        w_true = y_true[:, None, :, 2]                                          # (B, 1, K)
        frac = torch.ones_like(w_true) if self.weighting == "uniform" else w_true
        d_t1 = (y_pred[:, :, 0:1] - y_true[:, None, :, 0]) ** 2                # (B, Q, K)
        d_t2 = (y_pred[:, :, 1:2] - y_true[:, None, :, 1]) ** 2
        d_w = (y_pred[:, :, 2:3] - w_true) ** 2
        cost = (self.lw["t1"] * d_t1 + self.lw["t2"] * d_t2) * frac + self.lw["w"] * d_w \
            + self.lw["exist"] * (1 - torch.sigmoid(y_pred[:, :, 3:4]))

        cost_np, n_np = cost.detach().cpu().numpy(), n_comp.cpu().numpy()
        b_idx, q_idx, k_idx = [], [], []
        for b in range(B):
            r, c = linear_sum_assignment(cost_np[b, :, :n_np[b]])
            b_idx += [b] * len(r); q_idx += list(r); k_idx += list(c)
        dev = y_pred.device
        b_idx, q_idx, k_idx = (torch.as_tensor(a, device=dev) for a in (b_idx, q_idx, k_idx))

        f = frac[b_idx, 0, k_idx]
        count = torch.zeros(B, device=dev).index_add_(0, b_idx, torch.ones_like(f)).clamp(min=1)
        per_voxel = lambda v: torch.zeros(B, device=dev).index_add_(0, b_idx, v) / count
        t1 = per_voxel(self.lw["t1"] * d_t1[b_idx, q_idx, k_idx] * f)
        t2 = per_voxel(self.lw["t2"] * d_t2[b_idx, q_idx, k_idx] * f)
        w = per_voxel(self.lw["w"] * d_w[b_idx, q_idx, k_idx])

        target = torch.zeros(B, Q, device=dev)
        target[b_idx, q_idx] = 1.0
        n_pos = target.sum(1, keepdim=True)
        pos_weight = torch.clamp((Q - n_pos) / n_pos.clamp(min=1), 0.5, 10.0)
        ex = self.lw["exist"] * F.binary_cross_entropy_with_logits(
            y_pred[:, :, 3], target, pos_weight=pos_weight, reduction="none").mean(1)
        return (t1 + t2 + w + ex).mean()
