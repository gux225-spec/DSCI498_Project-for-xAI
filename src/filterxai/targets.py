"""Explanation targets F(g) (algorithm_plan_v2.md §3), differentiable in torch.

All curve targets take dB curves (B, F, 2) = [S21_dB, S11_dB] on the fixed 601-point core grid,
so every surrogate (complex or dB output) is explained through the same function.
Smooth max/min use log-sum-exp with temperature TAU_DB; smooth relu uses TAU_DB * softplus(x / TAU_DB).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .data import region_masks

TAU_DB = 0.05
CENTROID_GHZ = (10.5, 13.5)
T1_S21_GHZ = (10.8, 11.6, 12.0, 12.4, 13.2)
T1_S11_GHZ = (12.0,)
# dataset's provisional mask (package_supplement_dataset.py): limits in dB
MASK = {"passband_S21_min": -0.5, "passband_S11_max": -10.0, "low_stopband_S21_max": -20.0, "high_stopband_S21_max": -15.0}

CURVE_TARGETS = (["T0_centroid_GHz"] + [f"T1_S21_{f}" for f in T1_S21_GHZ] + [f"T1_S11_{f}" for f in T1_S11_GHZ]
                 + ["T1b_passband_mean_S21_dB", "T2_ripple_dB", "T2_worst_S11_dB", "T2_mask_violation_dB"])
CONCEPT_TARGETS = ["T3_f2_GHz", "T3_f3_GHz", "T3_f4_GHz", "T3_Q2", "T3_Q3", "T3_Q4"]  # resonator poles 2-4
GROUPS = {"T0": ["T0_centroid_GHz"], "T1": [t for t in CURVE_TARGETS if t.startswith("T1_")],
          "T1b": ["T1b_passband_mean_S21_dB"], "T2": [t for t in CURVE_TARGETS if t.startswith("T2_")],
          "T3": CONCEPT_TARGETS}


def smax(x, tau=TAU_DB):
    return tau * torch.logsumexp(x / tau, dim=-1)


def srelu(x, tau=TAU_DB):
    return tau * F.softplus(x / tau)


class CurveTargets:
    def __init__(self, f_ghz, device="cpu"):
        f = np.asarray(f_ghz)
        r = region_masks(f)
        t = lambda m: torch.tensor(np.flatnonzero(m), device=device)  # noqa: E731
        self.pb, self.low, self.high = t(r["passband"]), t(r["low_stopband"]), t(r["high_stopband"])
        self.cen = t((f >= CENTROID_GHZ[0] - 1e-9) & (f <= CENTROID_GHZ[1] + 1e-9))
        self.f_cen = torch.tensor(f[self.cen.cpu().numpy()], dtype=torch.float32, device=device)
        self.i21 = torch.tensor([int(np.argmin(np.abs(f - g))) for g in T1_S21_GHZ], device=device)
        self.i11 = torch.tensor([int(np.argmin(np.abs(f - g))) for g in T1_S11_GHZ], device=device)
        self.names = list(CURVE_TARGETS)

    def __call__(self, db):
        s21, s11 = db[..., 0], db[..., 1]
        w = 10 ** (s21[:, self.cen] / 10)
        centroid = (w * self.f_cen).sum(-1) / w.sum(-1)
        p21, p11 = s21[:, self.pb], s11[:, self.pb]
        ripple = smax(p21) + smax(-p21)
        worst11 = smax(p11)
        viol = (srelu(MASK["passband_S21_min"] + smax(-p21)) + srelu(worst11 - MASK["passband_S11_max"])
                + srelu(smax(s21[:, self.low]) - MASK["low_stopband_S21_max"])
                + srelu(smax(s21[:, self.high]) - MASK["high_stopband_S21_max"]))
        return torch.cat([centroid[:, None], s21[:, self.i21], s11[:, self.i11], p21.mean(-1, keepdim=True),
                          ripple[:, None], worst11[:, None], viol[:, None]], dim=1)


def hard_targets(f_ghz, db):
    """Non-smooth counterparts (numpy) of the T2 targets, for checking the smoothing."""
    r = region_masks(np.asarray(f_ghz))
    p21, p11 = db[:, r["passband"], 0], db[:, r["passband"], 1]
    viol = (np.maximum(0, MASK["passband_S21_min"] - p21.min(1)) + np.maximum(0, p11.max(1) - MASK["passband_S11_max"])
            + np.maximum(0, db[:, r["low_stopband"], 0].max(1) - MASK["low_stopband_S21_max"])
            + np.maximum(0, db[:, r["high_stopband"], 0].max(1) - MASK["high_stopband_S21_max"]))
    return {"T2_ripple_dB": p21.max(1) - p21.min(1), "T2_worst_S11_dB": p11.max(1), "T2_mask_violation_dB": viol}
