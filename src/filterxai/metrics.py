"""Validation metrics against the gate in algorithm_plan_v2.md §2.5.

Band edges follow the dataset convention (docs/reference/baseline_review.md): threshold
max(S21_dB) - 3 dB, the connected interval around the sampled peak, linear interpolation.
"""
import numpy as np

from .data import region_masks

GOALS = {
    "S21_full_MAE_dB": 0.5,
    "S21_passband_MAE_dB": 0.2,
    "S11_full_MAE_dB": 2.0,
    "S11_MAE_true_ge_m30dB": 1.0,
    "edge_MAE_MHz": 10.0,
    "edge_P95_MHz": 30.0,
}
DB_FLOOR = -300.0


def to_db(S):
    return np.maximum(20 * np.log10(np.maximum(np.abs(S), 1e-300)), DB_FLOOR)


def band_edges(f_ghz, s21_db):
    """(f_lower, f_upper) of the relative -3 dB interval containing the sampled peak; NaN where clipped."""
    n = len(s21_db)
    out = np.full((n, 2), np.nan)
    for i, y in enumerate(s21_db):
        k = int(np.argmax(y))
        thr = y[k] - 3.0
        lo = k
        while lo > 0 and y[lo - 1] >= thr:
            lo -= 1
        hi = k
        while hi < len(y) - 1 and y[hi + 1] >= thr:
            hi += 1
        if lo > 0:
            out[i, 0] = np.interp(thr, [y[lo - 1], y[lo]], [f_ghz[lo - 1], f_ghz[lo]])
        if hi < len(y) - 1:
            out[i, 1] = np.interp(thr, [y[hi + 1], y[hi]], [f_ghz[hi + 1], f_ghz[hi]])
    return out


def evaluate(f_ghz, true_db, pred_db, single_band, groups=None, true_S=None, pred_S=None):
    """Gate metrics overall and per group. *_db are (N, F, 2) = [S21_dB, S11_dB]."""
    regions = region_masks(f_ghz)
    pb = regions["passband"]
    err = np.abs(pred_db - true_db)
    e_true = band_edges(f_ghz, true_db[..., 0])
    e_pred = band_edges(f_ghz, pred_db[..., 0])
    edge_err = np.abs(e_pred - e_true) * 1e3  # MHz, (N, 2)

    def block(m):
        sb = m & single_band
        ee = edge_err[sb].ravel()
        ee = ee[np.isfinite(ee)]
        s11_mask = true_db[m, :, 1] >= -30
        r = {
            "n_designs": int(m.sum()),
            "S21_full_MAE_dB": float(err[m, :, 0].mean()),
            "S21_passband_MAE_dB": float(err[m][:, pb, 0].mean()),
            "S11_full_MAE_dB": float(err[m, :, 1].mean()),
            "S11_MAE_true_ge_m30dB": float(err[m, :, 1][s11_mask].mean()),
            "edge_MAE_MHz": float(ee.mean()) if ee.size else float("nan"),
            "edge_P95_MHz": float(np.quantile(ee, 0.95)) if ee.size else float("nan"),
            "n_edges": int(ee.size),
            "S21_MAE_by_region_dB": {k: float(err[m][:, v, 0].mean()) for k, v in regions.items()},
            "S11_MAE_by_region_dB": {k: float(err[m][:, v, 1].mean()) for k, v in regions.items()},
        }
        if pred_S is not None:
            r["complex_RMSE"] = float(np.sqrt(np.mean(np.abs(pred_S[m] - true_S[m]) ** 2)))
            power = np.abs(pred_S[m, :, 0]) ** 2 + np.abs(pred_S[m, :, 1]) ** 2
            r["power_balance_max_abs"] = float(np.abs(power - 1).max())
            r["power_balance_mean_abs"] = float(np.abs(power - 1).mean())
        return r

    out = {"all": block(np.ones(len(true_db), bool))}
    for name, m in (groups or {}).items():
        out[name] = block(m)
    out["all"]["passes_gate"] = {k: bool(out["all"][k] <= g) for k, g in GOALS.items()}
    out["all"]["composite_score"] = float(np.mean([out["all"][k] / g for k, g in GOALS.items()]))
    return out


def standard_groups(d):
    return {
        "legacy_4d": d["source"] == "legacy_4d_actual",
        "new_6d": d["source"] == "new_6d_actual",
        "single_band": d["single_band"],
        "multiband": ~d["single_band"],
    }
