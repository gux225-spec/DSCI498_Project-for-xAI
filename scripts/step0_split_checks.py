"""Step 0: pipeline and split checks. Reads test inputs and ids only, never test labels.

Writes results/step0/split_checks.json and exits non-zero if any check fails.
Usage: python scripts/step0_split_checks.py
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data  # noqa: E402

out_dir = data.ROOT / "results" / "step0"
out_dir.mkdir(parents=True, exist_ok=True)

tr, va, te = data.load("train"), data.load("validation"), data.load("test")
checks = {}


def check(name, ok, **info):
    checks[name] = {"ok": bool(ok), **info}


check("counts", (len(tr["X"]), len(va["X"]), len(te["X"])) == (1008, 216, 216),
      train=len(tr["X"]), validation=len(va["X"]), test=len(te["X"]))
check("test_labels_locked", te["labels_locked"] and "S" not in te)
ids = [set(d["design_id"].tolist()) for d in (tr, va, te)]
check("design_ids_disjoint", not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]))
check("design_ids_unique", all(len(s) == len(d["design_id"]) for s, d in zip(ids, (tr, va, te))))

geo = [np.round(d["X_raw"][:, :6], 6) for d in (tr, va, te)]
rows = [set(map(tuple, g)) for g in geo]
check("geometry_no_cross_split_duplicates", not (rows[0] & rows[1] or rows[0] & rows[2] or rows[1] & rows[2]),
      train_va=len(rows[0] & rows[1]), train_te=len(rows[0] & rows[2]), va_te=len(rows[1] & rows[2]))

f = tr["frequency_GHz"]
check("frequency_grid", len(f) == 601 and np.isclose(f[0], 9.0) and np.isclose(f[-1], 15.0)
      and np.allclose(np.diff(f), 0.01) and np.array_equal(f, va["frequency_GHz"]) and np.array_equal(f, te["frequency_GHz"]))
check("feature_order", tr["feature_names"] == data.GEOMETRY + ["control_uniform", "control_normal", "control_unused_marker_mm"],
      names=tr["feature_names"])

mu, sd = tr["X"].mean(0), tr["X"].std(0)
check("train_normalization", np.allclose(mu, 0, atol=1e-5) and np.allclose(sd, 1, atol=1e-4),
      mean_absmax=float(np.abs(mu).max()), std_range=[float(sd.min()), float(sd.max())])
check("finite", all(np.isfinite(d["X"]).all() for d in (tr, va, te)) and all(np.isfinite(d["S"]).all() for d in (tr, va)))

for name, d in (("train", tr), ("validation", va)):
    S = d["S"]
    db = 20 * np.log10(np.abs(S))
    check(f"{name}_dB_matches_complex", np.allclose(db, d["Y_dB"], atol=1e-6), max_err=float(np.abs(db - d["Y_dB"]).max()))
    power = np.abs(S[..., 0]) ** 2 + np.abs(S[..., 1]) ** 2
    check(f"{name}_port1_power_balance", np.abs(power - 1).max() < 1e-6, max_abs=float(np.abs(power - 1).max()))

masks = data.region_masks(f)
check("regions", sum(int(m.sum()) for m in masks.values()) == 601, sizes={k: int(m.sum()) for k, m in masks.items()})

ref = (data.REFERENCE_GEOMETRY_MIL - np.array(data.scalers()["geometry"]["mean"])) / np.array(data.scalers()["geometry"]["effective_std"])
check("reference_point_normalized", np.abs(ref).max() < 0.01, z=ref.round(4).tolist())

groups = {}
for name, d in (("train", tr), ("validation", va)):
    groups[name] = {"source": {s: int((d["source"] == s).sum()) for s in np.unique(d["source"])},
                    "multiband": int((~d["single_band"]).sum()), "mask_pass": int(d["mask_pass"].sum())}
groups["test"] = {"source": {s: int((te["source"] == s).sum()) for s in np.unique(te["source"])}}

report = {"all_ok": all(c["ok"] for c in checks.values()), "checks": checks, "groups": groups}
(out_dir / "split_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
for k, c in checks.items():
    print(("OK  " if c["ok"] else "FAIL"), k, {a: b for a, b in c.items() if a != "ok"} or "")
print(json.dumps(groups))
sys.exit(0 if report["all_ok"] else 1)
