"""Independent read-only audit of the v2 release arrays (NumPy only).

Usage: python audit_release.py <release_dir>
"""
import sys, csv
import numpy as np

rel = sys.argv[1]
D = rel + r"\data"


def load(name):
    z = np.load(f"{D}\\{name}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


core = {s: load(f"core_{s}") for s in ("train", "validation", "test")}
wide = {s: load(f"wideband_{s}") for s in ("train", "validation", "test")}

print("== keys/shapes core_train")
for k, v in core["train"].items():
    print(f"  {k:45s} {str(v.shape):22s} {v.dtype}")
print("== keys wideband_train")
for k, v in wide["train"].items():
    print(f"  {k:45s} {str(v.shape):22s} {v.dtype}")


def ids(d):
    for k in ("design_id", "design_ids"):
        if k in d:
            return [str(x) for x in d[k]]
    return None


# split leakage: design ids and geometry rows
allg = {}
for s in core:
    g = core[s]["geometry_mil"]
    allg[s] = {tuple(np.round(r, 9)) for r in g}
    print(f"{s}: n={len(g)} unique_geom={len(allg[s])}")
    i = ids(core[s])
    if i:
        print(f"   unique ids={len(set(i))}")
for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
    print(f"geom overlap {a}/{b}: {len(allg[a] & allg[b])}")
    ia, ib = ids(core[a]), ids(core[b])
    if ia and ib:
        print(f"id overlap {a}/{b}: {len(set(ia) & set(ib))}")

names = ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]
for s in core:
    g = core[s]["geometry_mil"]
    print(s, " ".join(f"{n}[{g[:, j].min():.3f},{g[:, j].max():.3f}]" for j, n in enumerate(names)))

# numeric sanity + lossless-circuit physics on core
for s in core:
    S = core[s]["S_complex"]
    Y = core[s]["Y_dB"]
    fin = np.isfinite(S).all() and np.isfinite(Y).all()
    S21, S11 = S[:, :, 1, 0], S[:, :, 0, 0]
    yd = np.max(np.abs(20 * np.log10(np.maximum(np.abs(S21), 1e-15)) - Y[..., 0]))
    yd11 = np.max(np.abs(20 * np.log10(np.maximum(np.abs(S11), 1e-15)) - Y[..., 1]))
    recip = np.max(np.abs(S[:, :, 0, 1] - S[:, :, 1, 0]))
    power = np.max(np.abs(np.abs(S11) ** 2 + np.abs(S21) ** 2 - 1))
    print(f"{s}: finite={fin} |Y-20log|S21||max={yd:.2e} S11={yd11:.2e} reciprocity={recip:.2e} "
          f"|S11|^2+|S21|^2-1 max={power:.2e} S21dB [{Y[..., 0].min():.1f},{Y[..., 0].max():.4f}] "
          f"S11dB min {Y[..., 1].min():.1f}")

f = core["train"]["frequency_Hz"]
print("core freq", f.shape, f.min(), f.max(), np.unique(np.round(np.diff(f))))
fw = wide["train"]["frequency_Hz"]
print("wide freq", fw.shape, fw.min(), fw.max(), np.unique(np.round(np.diff(fw))))

for s in core:
    same_geom = np.allclose(core[s]["geometry_mil"], wide[s]["geometry_mil"])
    err = np.max(np.abs(core[s]["S_complex"] - wide[s]["S_complex"][:, :601]))
    print(f"{s}: core/wide same geometry order={same_geom} shared-S max err={err:.2e}")

Yt = core["train"]["Y_dB"][..., 0]
print("S21 curves exact-duplicate count:", len(Yt) - len({r.tobytes() for r in Yt}))

X = core["train"]["X_candidate"]
with open(f"{D}\\candidate_inputs.csv", newline="", encoding="utf-8") as fh:
    cols = next(csv.reader(fh))
print("candidate_inputs.csv header:", cols)
print("X_candidate per-column std (train):", np.round(X.std(0), 6).tolist())

with open(f"{D}\\design_metrics.csv", newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
print("design_metrics rows", len(rows), "cols", len(rows[0]))
print("  columns:", list(rows[0].keys()))
for key in rows[0]:
    kl = key.lower()
    if any(t in kl for t in ("multi", "status", "n_pass", "mask_pass", "source", "split")):
        vals = {}
        for r in rows:
            vals[r[key]] = vals.get(r[key], 0) + 1
        if len(vals) < 15:
            print(f"  {key}: {vals}")

for s in core:
    i = ids(core[s])
    if i:
        pref = {}
        for x in i:
            p = x.rsplit("_", 1)[0]
            pref[p] = pref.get(p, 0) + 1
        print(s, pref)
