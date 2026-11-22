"""openEMS full-wave model of the coupled-line band-pass filter (layout = Momentum template).

The five conductor outlines come from the Momentum diagnostic template
(D:/Downloads/ADS_XAI_runs/scripts/run_momentum_diagnostics.py, polygons()), which was
checked against the native ADS layout export. Substrate and conductor follow ads/baseline.ckt:
H = 15 mil, Er = 3.78, lossless, ideal conductor (zero-thickness PEC sheets here).

Not a replacement for HFSS or measurement: open-boundary FDTD with a finite grid. The
baseline is compared against Momentum before any other case is interpreted.

Run with E:/anaconda3/envs/openems/python.exe  (needs CSXCAD_INSTALL_PATH=E:/openEMS).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("CSXCAD_INSTALL_PATH", r"E:\openEMS")
from CSXCAD import ContinuousStructure  # noqa: E402
from openEMS import openEMS  # noqa: E402

MIL = 25.4e-6
FIXED = {"Lfeed": 100.0, "Er": 3.78, "H": 15.0, "Wfeed": 32.0}
FREQ_HZ = np.linspace(9e9, 15e9, 601)  # same grid as the ADS core data

# Mesh settings (mil). Version tag goes into every result file.
MESH = {"version": 1, "max_cell_metal": 6.0, "max_cell_far": 25.0, "gap_cells": 3,
        "merge_tol": 1.0, "sub_cells": 4, "air_top": 250.0, "y_margin": 150.0,
        "port_ext": 200.0, "end_criteria": 1e-4, "max_ts": 400000}


def _pair_outline(a, b):
    x0, x1, y0, y1 = a
    xx0, x2, yy0, yy1 = b
    assert abs(x1 - xx0) < 1e-9 and min(y1, yy1) > max(y0, yy0)
    return [(x1, y1), (x0, y1), (x0, y0), (x1, y0), (x1, yy0), (x2, yy0), (x2, yy1), (x1, yy1)]


def polygons(p):
    """Five conductor outlines in mil; port 1 at x=0 (y=0), port 2 at (xr, yr). Same as Momentum template."""
    lf, l1, l2, w1, w2 = p["Lfeed"], p["L1"], p["L2"], p["Wc1"], p["Wc2"]
    pin_in, pin_out = 32.0, 31.5  # fixed PCell connection widths, as in the native layout
    x1 = lf + l1
    x2 = x1 + l2
    bottom1 = -pin_in / 2
    top1 = bottom1 + w1
    lower1top = bottom1 - p["S1"]
    pin1y = lower1top - pin_out / 2
    upper2bottom = pin1y - pin_in / 2
    upper2top = upper2bottom + w2
    lower2top = upper2bottom - p["S2"]
    lower2bottom = lower2top - w2
    pin2y = lower2top - pin_out / 2
    xr, yr = 2 * x2, 2 * pin2y
    one = [(x1, bottom1), (x1, top1), (lf, top1), (lf, 16.0), (0.0, 16.0), (0.0, 0.0), (0.0, -16.0)]
    two = _pair_outline((lf, x1, lower1top - w1, lower1top), (x1, x2, upper2bottom, upper2top))
    three = _pair_outline((x1, x2, lower2bottom, lower2top), (x2, xr - x1, yr - lower2top, yr - lower2bottom))
    mirror = lambda poly: [(xr - x, yr - y) for x, y in poly]
    return [one, two, three, mirror(two), mirror(one)], (xr, yr)


def _merge(vals, tol):
    vals = np.sort(np.asarray(vals, float))
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] > tol:
            out.append(v)
    return np.array(out)


def _axis_lines(edges, gap_cells, tol):
    """Edge lines plus extra lines inside narrow gaps/strips so each has >= gap_cells cells."""
    e = _merge(edges, tol)
    lines = list(e)
    for a, b in zip(e[:-1], e[1:]):
        if b - a < 4 * gap_cells:  # narrow region: subdivide uniformly
            lines += list(np.linspace(a, b, gap_cells + 1)[1:-1])
    return np.array(lines)


def simulate(geom, sim_dir, threads=0, verbose=1, mesh_overrides=None):
    """Run one design. geom: dict with L1, L2, S1, S2, Wc1, Wc2 (mil). Returns dict with f, S, timing."""
    global MESH
    saved = MESH
    MESH = dict(MESH, **(mesh_overrides or {}))
    try:
        out = _simulate(geom, sim_dir, threads, verbose)
    finally:
        out_mesh, MESH = MESH, saved
    out["mesh"] = out_mesh
    return out


def _simulate(geom, sim_dir, threads, verbose):
    p = dict(FIXED, **{k: float(geom[k]) for k in ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]})
    polys, (xr, yr) = polygons(p)
    H, er, ext = p["H"], p["Er"], MESH["port_ext"]

    fdtd = openEMS(NrTS=MESH["max_ts"], EndCriteria=MESH["end_criteria"])
    fdtd.SetGaussExcite(12e9, 4e9)
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(MIL)

    xs = [x for poly in polys for x, _ in poly]
    ys = [y for poly in polys for _, y in poly] + [yr - 16, yr + 16]
    ymin, ymax = min(ys), max(ys)
    x_lines = _axis_lines(xs, MESH["gap_cells"], MESH["merge_tol"])
    y_lines = _axis_lines(ys, MESH["gap_cells"], MESH["merge_tol"])
    mesh.AddLine("x", x_lines)
    mesh.AddLine("y", y_lines)
    mesh.SmoothMeshLines("x", MESH["max_cell_metal"], 1.4)
    mesh.SmoothMeshLines("y", MESH["max_cell_metal"], 1.4)
    mesh.AddLine("x", [-ext, xr + ext])
    mesh.AddLine("y", [ymin - MESH["y_margin"], ymax + MESH["y_margin"]])
    mesh.SmoothMeshLines("x", MESH["max_cell_far"], 1.4)
    mesh.SmoothMeshLines("y", MESH["max_cell_far"], 1.4)
    mesh.AddLine("z", np.linspace(0, H, MESH["sub_cells"] + 1))
    mesh.AddLine("z", MESH["air_top"])
    mesh.SmoothMeshLines("z", MESH["max_cell_far"], 1.4)

    sub = csx.AddMaterial("substrate", epsilon=er)
    sub.AddBox([-ext, ymin - MESH["y_margin"], 0], [xr + ext, ymax + MESH["y_margin"], H], priority=0)

    pec = csx.AddMetal("PEC")
    for poly in polys:
        pts = np.array(poly).T  # 2 x N
        pec.AddPolygon(pts, "z", H, priority=10)

    w = FIXED["Wfeed"] / 2
    feed_shift, meas_shift = 80.0, 140.0
    p1 = fdtd.AddMSLPort(1, pec, [-ext, -w, H], [0, w, 0], "x", "z", excite=-1,
                         FeedShift=feed_shift, MeasPlaneShift=meas_shift, priority=10)
    p2 = fdtd.AddMSLPort(2, pec, [xr + ext, yr - w, H], [xr, yr + w, 0], "x", "z",
                         MeasPlaneShift=meas_shift, priority=10)

    n_cells = [len(mesh.GetLines(a)) for a in "xyz"]
    sim_dir = Path(sim_dir).resolve()  # Run() changes cwd, so relative paths break CalcPort
    sim_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cwd = os.getcwd()
    try:
        fdtd.Run(str(sim_dir), cleanup=True, verbose=verbose, numThreads=threads)
    finally:
        os.chdir(cwd)  # Run() leaves the process inside sim_dir
    elapsed = time.time() - t0

    f = FREQ_HZ
    for port in (p1, p2):
        port.CalcPort(str(sim_dir), f, ref_impedance=50)
    s11 = p1.uf_ref / p1.uf_inc
    s21 = p2.uf_ref / p1.uf_inc
    # Layout is point-symmetric (mirror about the centre) and reciprocal: S22 = S11, S12 = S21.
    S = np.zeros((len(f), 2, 2), complex)
    S[:, 0, 0] = S[:, 1, 1] = s11
    S[:, 1, 0] = S[:, 0, 1] = s21
    return {"frequency_Hz": f, "S_complex": S, "elapsed_s": elapsed, "mesh_lines_xyz": n_cells,
            "geometry": p}


def band_metrics(f_hz, s21_db, level=-3.0):
    """Arithmetic centre and width of the contiguous S21 > level band containing the S21 maximum."""
    f = np.asarray(f_hz) / 1e9
    above = s21_db > level
    k = int(np.argmax(s21_db))
    if not above[k]:
        return {"f0_GHz": np.nan, "BW_GHz": np.nan, "peak_S21_dB": float(s21_db[k])}
    lo, hi = k, k
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(f) - 1 and above[hi + 1]:
        hi += 1

    def cross(i, j):  # linear interpolation of the level crossing between samples i and j
        return f[i] + (level - s21_db[i]) * (f[j] - f[i]) / (s21_db[j] - s21_db[i])

    f_lo = cross(lo - 1, lo) if lo > 0 else f[0]
    f_hi = cross(hi, hi + 1) if hi < len(f) - 1 else f[-1]
    return {"f0_GHz": float((f_lo + f_hi) / 2), "BW_GHz": float(f_hi - f_lo), "peak_S21_dB": float(s21_db[k])}


def to_db(S):
    """Y_dB layout used by the dataset: [..., 0] = S21 dB, [..., 1] = S11 dB."""
    return np.stack([20 * np.log10(np.abs(S[..., 1, 0])), 20 * np.log10(np.abs(S[..., 0, 0]))], -1)


def mask_violation(f_hz, y_db):
    """Hard mask violation (dB) and its four components, same definition as src/filterxai/targets.py.

    MASK / REGIONS_GHZ are copied from targets.py / data.py (those import torch, absent in this env).
    """
    MASK = {"passband_S21_min": -0.5, "passband_S11_max": -10.0, "low_stopband_S21_max": -20.0,
            "high_stopband_S21_max": -15.0}
    REGIONS_GHZ = {"low_stopband": (9.0, 10.8), "passband": (11.6, 12.4), "high_stopband": (13.2, 15.0)}
    fg = np.asarray(f_hz) / 1e9
    r = {k: (fg >= lo - 1e-9) & (fg <= hi + 1e-9) for k, (lo, hi) in REGIONS_GHZ.items()}
    comps = {
        "passband_S21": max(0.0, MASK["passband_S21_min"] - y_db[r["passband"], 0].min()),
        "passband_S11": max(0.0, y_db[r["passband"], 1].max() - MASK["passband_S11_max"]),
        "low_stopband_S21": max(0.0, y_db[r["low_stopband"], 0].max() - MASK["low_stopband_S21_max"]),
        "high_stopband_S21": max(0.0, y_db[r["high_stopband"], 0].max() - MASK["high_stopband_S21_max"]),
    }
    return float(sum(comps.values())), {k: float(v) for k, v in comps.items()}


if __name__ == "__main__":
    # quick single run: python scripts/oems_filter.py <out_dir> [L1 L2 S1 S2 Wc1 Wc2]
    out = Path(sys.argv[1])
    vals = [float(v) for v in sys.argv[2:8]] or [138.5, 136.0, 3.3, 18.0, 25.5, 31.5]
    geom = dict(zip(["L1", "L2", "S1", "S2", "Wc1", "Wc2"], vals))
    res = simulate(geom, out / "sim")
    np.savez(out / "response.npz", frequency_Hz=res["frequency_Hz"], S_complex=res["S_complex"])
    ydb = to_db(res["S_complex"])
    v, comps = mask_violation(res["frequency_Hz"], ydb)
    summary = {"geometry_mil": geom, "elapsed_s": res["elapsed_s"], "mesh_lines_xyz": res["mesh_lines_xyz"],
               "mesh": MESH, **band_metrics(res["frequency_Hz"], ydb[:, 0]),
               "mask_violation_dB": v, "mask_components_dB": comps}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
