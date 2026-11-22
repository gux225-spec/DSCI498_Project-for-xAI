"""ADS circuit simulation of geometry jobs on the 9-15 GHz core grid (601 points). Run with ADS Python:

  E:\\ADS2026\\tools\\python\\python.exe scripts/ads_sim.py <jobs.json> <out_dir> [--workers N]

jobs.json: [{"id": str, "geometry": {"L1": mil, ..., "Wc2": mil}}, ...]. Every other physical parameter
stays at the dataset baseline. The netlist is patched exactly as in the dataset generator
(data/filter_supplement_v2/scripts/supplement_filter_dataset.py: patch, read_ds) from ads/baseline.ckt,
whose SHA256 matches plan.json source_snapshot_sha256. Each finished job is written to
<out_dir>/cases/<id>.npz (S_complex (601, 2, 2), output port then input port); existing files are skipped,
so the run is resumable. Failures are logged to <out_dir>/failures.jsonl and retried once.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ADS = Path("E:/ADS2026")
NETLIST = ROOT / "ads" / "baseline.ckt"
SHA = "f61c3c7d0dbca0764be613fceaf89a3b58dc6e6c3e6d75d0ae4b0cf417634a35"
STOP, STEP = 15.0, 0.01
BASE = {"L1": 138.5, "L2": 136.0, "S1": 3.3, "S2": 18.0, "Wc1": 25.5, "Wc2": 31.5}


def patch(source, g):
    """Same regex patching as the dataset generator, geometry only (feed and substrate stay at baseline)."""
    p = {**BASE, **g}
    text = source
    specs = [("MCFIL:CLin1", {"L": p["L1"], "S": p["S1"], "W": p["Wc1"]}),
             ("MCFIL:CLin2", {"L": p["L2"], "S": p["S2"], "W": p["Wc2"]})]
    for prefix, changes in specs:
        hits = list(re.finditer(rf"(?m)^{re.escape(prefix)}\s+[^\r\n]*$", text))
        if len(hits) != 1:
            raise ValueError("Ambiguous component: " + prefix)
        line = hits[0].group(0)
        for key, value in changes.items():
            token = rf"(?<!\w){key}=([-+\d.eE]+) mil(?=\s|$)"
            line, n = re.subn(token, lambda m: f"{key}={value:.12g} mil", line)
            if n != 1 or not np.isclose(float(re.search(token, line)[1]), value, rtol=1e-11, atol=1e-11):
                raise ValueError(f"Cannot set {prefix}.{key}")
        text = text[:hits[0].start()] + line + text[hits[0].end():]
    text, n = re.subn(r"(?m)^SweepPlan: SP1_stim Start=[^\r\n]+$",
                      f"SweepPlan: SP1_stim Start=9.0 GHz Stop={STOP:.12g} GHz Step={STEP:.12g} GHz ", text)
    if n != 1:
        raise ValueError("Ambiguous sweep statement")
    return text


def read_ds(path):
    from keysight.ads import dataset
    with dataset.open(path) as ds:
        blocks = [k for k in ds.varblock_names if {"S[1,1]", "S[1,2]", "S[2,1]", "S[2,2]"}.issubset({v.name for v in ds[k].dvars})]
        if len(blocks) != 1:
            raise ValueError("Ambiguous S parameter block")
        frame = ds[blocks[0]].to_dataframe().reset_index()
    f = frame["freq"].to_numpy(float)
    expected = np.linspace(9e9, STOP * 1e9, round((STOP - 9) / STEP) + 1)
    if f.shape != expected.shape or not np.allclose(f, expected, rtol=0, atol=1):
        raise ValueError("Sweep grid mismatch")
    s = np.empty((len(f), 2, 2), complex)
    for i in range(2):
        for j in range(2):
            s[:, i, j] = frame[f"S[{i + 1},{j + 1}]"].to_numpy(complex)
    if not np.isfinite(s).all():
        raise ValueError("Nonfinite S data")
    return f, s


def worker(args):
    jobs, out_dir, work_root = args
    os.environ["HPEESOF_DIR"] = str(ADS)
    from keysight.edatoolbox.ads import CircuitSimulator
    sim = CircuitSimulator(hpeesof_dir=str(ADS))
    source = NETLIST.read_text(encoding="utf-8")
    done = 0
    for job in jobs:
        out = Path(out_dir) / "cases" / f"{job['id']}.npz"
        if out.exists():
            continue
        for attempt in (1, 2):
            case = Path(tempfile.mkdtemp(prefix="c_", dir=work_root))
            try:
                sim.run_netlist(patch(source, job["geometry"]), output_dir=str(case), working_dir=str(case),
                                netlist_file=str(case / "design.ckt"), output_file=str(case / "simulator.log"),
                                dataset_name="response")
                log = (case / "simulator.log").read_text(encoding="utf-8", errors="replace")
                ds = list(case.glob("*.ds"))
                if len(ds) != 1 or "Simulation finished" not in log or "Error detected by hpeesofsim" in log:
                    raise ValueError("Native solver completion not confirmed")
                f, s = read_ds(ds[0])
                tmp = out.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, S_complex=s, frequency_Hz=f, geometry=json.dumps({**BASE, **job["geometry"]}))
                tmp.replace(out)
                done += 1
                break
            except Exception as e:  # noqa: BLE001
                with open(Path(out_dir) / "failures.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"id": job["id"], "attempt": attempt, "error": f"{type(e).__name__}: {e}"}) + "\n")
            finally:
                shutil.rmtree(case, ignore_errors=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if hashlib.sha256(NETLIST.read_bytes()).hexdigest() != SHA:
        raise RuntimeError("baseline netlist hash mismatch")
    jobs = json.loads(Path(a.jobs).read_text(encoding="utf-8"))
    out = Path(a.out_dir)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    todo = [j for j in jobs if not (out / "cases" / f"{j['id']}.npz").exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run, {a.workers} workers", flush=True)
    t0 = time.time()
    chunks = [(todo[i::a.workers], str(out), str(work)) for i in range(a.workers)]
    if a.workers == 1:
        n = worker(chunks[0])
    else:
        with mp.Pool(a.workers) as pool:
            n = sum(pool.map(worker, chunks))
    print(f"finished {n} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
