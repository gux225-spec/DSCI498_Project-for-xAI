"""How many shared poles does a rational (pole-residue) model need on this data? TRAIN only.

Fits S21 and S11 of each sampled train design with common poles (vector fitting,
Gustavsen & Semlyen 1999, real-valued formulation) and reports the fit error per order.

Usage: python vf_order_check.py <release_dir> [n_designs]
"""
import sys
import numpy as np

rel = sys.argv[1]
n_designs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
z = np.load(rel + r"\data\core_train.npz", allow_pickle=False)
f = z["frequency_Hz"] / 1e9
s = 2j * np.pi * f
S = z["S_complex"]


def basis(poles):
    cols = []
    k = 0
    while k < len(poles):
        a = poles[k]
        if abs(a.imag) > 0:
            cols += [1 / (s - a) + 1 / (s - a.conjugate()), 1j / (s - a) - 1j / (s - a.conjugate())]
            k += 2
        else:
            cols.append(1 / (s - a))
            k += 1
    return np.stack(cols, 1)


def initial_poles(n):
    beta = 2 * np.pi * np.linspace(f.min(), f.max(), n // 2)
    p = []
    for b in beta:
        p += [-b / 100 + 1j * b, -b / 100 - 1j * b]
    return np.array(p)


def realify(A, rhs):
    return np.vstack([A.real, A.imag]), np.concatenate([rhs.real, rhs.imag])


def vector_fit(H, n, iters=15):
    poles = initial_poles(n)
    M = H.shape[1]
    for _ in range(iters):
        Phi = basis(poles)
        N = Phi.shape[1]
        rows, rhs = [], []
        for m in range(M):
            block = np.zeros((len(s), M * (N + 1) + N), dtype=complex)
            block[:, m * (N + 1):m * (N + 1) + N] = Phi
            block[:, m * (N + 1) + N] = 1
            block[:, M * (N + 1):] = -H[:, m:m + 1] * Phi
            Ar, br = realify(block, H[:, m])
            rows.append(Ar)
            rhs.append(br)
        x, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
        ct = x[M * (N + 1):]
        A = np.zeros((N, N))
        b = np.zeros(N)
        k = 0
        while k < N:
            a = poles[k]
            A[k:k + 2, k:k + 2] = [[a.real, a.imag], [-a.imag, a.real]]
            b[k] = 2
            k += 2
        new = np.linalg.eigvals(A - np.outer(b, ct))
        new = np.where(new.real > 0, -new.real + 1j * new.imag, new)
        new = new[np.argsort(new.imag)]
        pos = new[new.imag > 0]
        poles = np.ravel(np.column_stack([pos, pos.conjugate()]))
    Phi = basis(poles)
    fits = []
    for m in range(M):
        Ar, br = realify(np.c_[Phi, np.ones(len(s))], H[:, m])
        c, *_ = np.linalg.lstsq(Ar, br, rcond=None)
        fits.append(np.c_[Phi, np.ones(len(s))] @ c)
    return poles, np.stack(fits, 1)


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-15))


rng = np.random.default_rng(2026)
idx = rng.choice(len(S), n_designs, replace=False)
pb = (f >= 11.6) & (f <= 12.4)
print(f"designs={n_designs} (train, seed 2026); shared poles for S21 and S11")
for n in (4, 6, 8, 10, 12, 14, 16):
    cmax, s21mae, s11pb, s11all, unstable = [], [], [], [], 0
    for i in idx:
        H = np.stack([S[i, :, 1, 0], S[i, :, 0, 0]], 1)
        poles, Hf = vector_fit(H, n)
        unstable += int(np.any(poles.real >= 0))
        cmax.append(np.abs(Hf - H).max())
        s21mae.append(np.abs(db(Hf[:, 0]) - db(H[:, 0])).mean())
        s11pb.append(np.abs(db(Hf[pb, 1]) - db(H[pb, 1])).mean())
        s11all.append(np.abs(db(Hf[:, 1]) - db(H[:, 1])).mean())
    print(f"N={n:2d}: max|dS| median {np.median(cmax):.2e} worst {np.max(cmax):.2e} | "
          f"S21 dB MAE median {np.median(s21mae):.4f} | S11 dB MAE all {np.median(s11all):.4f} pass {np.median(s11pb):.4f} "
          f"worst-pass {np.max(s11pb):.3f} | unstable={unstable}")

# pole locations of a few designs at a chosen order, to see whether they look like resonances
for i in idx[:3]:
    H = np.stack([S[i, :, 1, 0], S[i, :, 0, 0]], 1)
    poles, _ = vector_fit(H, 10)
    pos = poles[poles.imag > 0]
    print(z["design_id"][i], "N=10 poles (f GHz, Q):",
          [(round(p.imag / 2 / np.pi, 3), round(abs(p) / (2 * abs(p.real)), 1)) for p in pos])
