"""Mesh-refinement study of the openEMS baseline against Momentum and ADS.

Only the baseline geometry is used, so no test-case result informs the mesh choice.
Usage: python scripts/oems_mesh_study.py [level ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "fullwave_openems" / "mesh_study"
BASE = {"L1": 138.5, "L2": 136.0, "S1": 3.3, "S2": 18.0, "Wc1": 25.5, "Wc2": 31.5}
LEVELS = {
    "m1": {"max_cell_metal": 6.0, "gap_cells": 3, "sub_cells": 4},
    "m2": {"max_cell_metal": 3.0, "gap_cells": 4, "sub_cells": 6},
    "m3": {"max_cell_metal": 2.0, "gap_cells": 6, "sub_cells": 8, "merge_tol": 0.5},
    "m4": {"max_cell_metal": 1.3, "gap_cells": 8, "sub_cells": 10, "merge_tol": 0.3},
}


def main(names):
    import oems_filter as o
    for name in names:
        d = OUT / name
        if not (d / "summary.json").exists():
            d.mkdir(parents=True, exist_ok=True)
            r = o.simulate(BASE, d / "sim", verbose=0, mesh_overrides=LEVELS[name])
            np.savez(d / "response.npz", frequency_Hz=r["frequency_Hz"], S_complex=r["S_complex"])
            y = o.to_db(r["S_complex"])
            v, comps = o.mask_violation(r["frequency_Hz"], y)
            s = {"level": name, "mesh": r["mesh"], "mesh_lines_xyz": r["mesh_lines_xyz"], "elapsed_s": r["elapsed_s"],
                 **o.band_metrics(r["frequency_Hz"], y[:, 0]), "mask_violation_dB": v, "mask_components_dB": comps}
            (d / "summary.json").write_text(json.dumps(s, indent=2))
        s = json.loads((d / "summary.json").read_text())
        print(f"{name}: lines {s['mesh_lines_xyz']}  {s['elapsed_s']:.0f}s  f0 {s['f0_GHz']:.4f} GHz  "
              f"BW {s['BW_GHz']:.4f} GHz  viol {s['mask_violation_dB']:.2f} dB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(LEVELS))
