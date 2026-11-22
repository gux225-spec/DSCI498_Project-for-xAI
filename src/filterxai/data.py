"""Data access for the v2 release (core band only).

Wraps the release's own loader so normalization stays train-only, adds per-design
metadata, and refuses to hand out test labels unless explicitly allowed (protocol:
no test labels before configuration freeze).
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data" / "filter_supplement_v2"
sys.path.insert(0, str(RELEASE / "scripts"))
from load_supplement_data import load_split  # noqa: E402

PRESET = "geometry_plus_negative_controls"
GEOMETRY = ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]
REFERENCE_GEOMETRY_MIL = np.array([138.5, 136.0, 3.3, 18.0, 25.5, 31.5])
REGIONS_GHZ = {"low_stopband": (9.0, 10.8), "passband": (11.6, 12.4), "high_stopband": (13.2, 15.0)}
FREEZE = ROOT / "results" / "step3" / "freeze.json"  # written by scripts/step3_freeze.py before any test label is read


class TestLabelsLocked(RuntimeError):
    pass


def region_masks(freq_ghz):
    """Boolean masks for the four loss/report regions; transition is the complement."""
    masks = {k: (freq_ghz >= lo - 1e-9) & (freq_ghz <= hi + 1e-9) for k, (lo, hi) in REGIONS_GHZ.items()}
    masks["transition"] = ~np.any(np.stack(list(masks.values())), axis=0)
    return masks


def _metadata(split):
    """source / band count / mask pass per design from design_metrics.csv (labels for test)."""
    with open(RELEASE / "data" / "design_metrics.csv", encoding="utf-8-sig", newline="") as fh:
        rows = {r["design_id"]: r for r in csv.DictReader(fh) if r["split"] == split}
    return rows


def _sources(split):
    """source column from candidate_inputs.csv (provenance, safe for every split)."""
    with open(RELEASE / "data" / "candidate_inputs.csv", encoding="utf-8-sig", newline="") as fh:
        return {r["design_id"]: r["source"] for r in csv.DictReader(fh) if r["split"] == split}


def _test_inputs(control_replicate, preset):
    """Test inputs only. Reads input keys of the npz lazily, so S/dB arrays are never decompressed.

    Mirrors load_split's selection and train-only normalization.
    """
    names = json.loads((RELEASE / "feature_roles.json").read_text(encoding="utf-8"))["input_presets"][preset]
    sc = scalers()
    with np.load(RELEASE / "data" / "core_test.npz", allow_pickle=False) as z:
        all_names = z["candidate_feature_names"].tolist()
        idx = [all_names.index(n) for n in names]
        x_raw = z["X_candidate"][:, idx].copy()
        means = np.array(sc["candidates"]["mean"])[idx]
        sds = np.array(sc["candidates"]["effective_std"])[idx]
        if control_replicate:
            nuisance = sc["negative_control_replicates"][str(control_replicate)]
            for j, n in enumerate(z["negative_control_names"].tolist()):
                if n in names:
                    k = names.index(n)
                    x_raw[:, k] = z["negative_controls_replicates"][control_replicate, :, j]
                    means[k], sds[k] = nuisance["mean"][j], nuisance["effective_std"][j]
        ids, freq = z["design_id"].copy(), z["frequency_Hz"].copy()
    src = _sources("test")
    return {"X": ((x_raw - means) / sds).astype(np.float32), "X_raw": x_raw, "feature_names": names,
            "design_id": ids, "frequency_GHz": freq / 1e9, "source": np.array([src[i] for i in ids]),
            "labels_locked": True}


def load(split, control_replicate=0, preset=PRESET, allow_test_labels=False):
    """Inputs (normalized and raw), complex S, dB and metadata for one split.

    S_complex is reordered to (N, F, 2) = [S21, S11]. For split='test' without
    allow_test_labels only inputs, ids and source are returned.
    """
    if split == "test" and not allow_test_labels:
        return _test_inputs(control_replicate, preset)
    if split == "test" and not FREEZE.exists():
        raise TestLabelsLocked("test labels need the configuration freeze record " + str(FREEZE))

    d = load_split(RELEASE, split, "core", preset, control_replicate, normalize=True)
    x_raw = load_split(RELEASE, split, "core", preset, control_replicate, normalize=False)["X_design"]
    S = d["S_complex"]
    s21_s11 = np.stack([S[:, :, 1, 0], S[:, :, 0, 0]], axis=-1)
    ids = d["design_id"]
    meta = _metadata(split)
    rows = [meta[i] for i in ids]
    return {
        "X": d["X_design"].astype(np.float32),
        "X_raw": x_raw,
        "feature_names": d["feature_names"],
        "design_id": ids,
        "frequency_GHz": d["frequency_Hz"] / 1e9,
        "S": s21_s11,
        "Y_dB": d["Y_dB"],
        "source": np.array([r["source"] for r in rows]),
        "n_bands": np.array([int(r["n_connected_3dB_bands"]) for r in rows]),
        "single_band": np.array([r["single_band_center_valid"] == "True" for r in rows]),
        "mask_pass": np.array([r["mask_pass_sampled_grid"] == "True" for r in rows]),
        "labels_locked": False,
    }


def scalers():
    return json.loads((RELEASE / "scalers_train_only.json").read_text(encoding="utf-8"))
