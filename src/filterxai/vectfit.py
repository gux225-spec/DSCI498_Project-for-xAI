"""Vector fitting (Gustavsen & Semlyen 1999, real formulation) with poles shared across channels.

Model per channel c:  H_c(s) = d_c + sum_k r_ck / (s - p_k),  s = j*2*pi*f, f in GHz.
Complex poles appear as conjugate pairs with conjugate residues, so H is the response of
a real system. Same algorithm as scripts/vf_order_check.py, generalized to real poles and
returning residues.
"""
import numpy as np


def _basis(s, poles):
    """Real-formulation basis; complex poles must be stored as (a, conj(a)) adjacent pairs."""
    cols, k = [], 0
    while k < len(poles):
        a = poles[k]
        if a.imag != 0:
            cols += [1 / (s - a) + 1 / (s - a.conjugate()), 1j / (s - a) - 1j / (s - a.conjugate())]
            k += 2
        else:
            cols.append(1 / (s - a))
            k += 1
    return np.stack(cols, 1)


def _arrange(new):
    """Flip unstable poles, then store complex ones as (a, conj a) pairs sorted by |a|, real ones after."""
    new = np.where(new.real > 0, -new.real + 1j * new.imag, new)
    tol = 1e-9 * np.abs(new).max()
    cplx = new[new.imag > tol]
    real = new[np.abs(new.imag) <= tol].real
    cplx = cplx[np.argsort(np.abs(cplx))]
    return np.concatenate([np.ravel(np.column_stack([cplx, cplx.conjugate()])), np.sort(real).astype(complex)])


def initial_poles(f_ghz, n, damping=0.01):
    beta = 2 * np.pi * np.linspace(f_ghz.min(), f_ghz.max(), n // 2)
    return np.ravel(np.column_stack([-damping * beta + 1j * beta, -damping * beta - 1j * beta]))


def _realify(A, b):
    return np.vstack([A.real, A.imag]), np.concatenate([b.real, b.imag])


def vector_fit(f_ghz, H, n=10, iters=15, poles=None):
    """Fit H (F, M) with n shared poles. Returns poles, residues (M, n), constants (M,), fitted H."""
    s = 2j * np.pi * f_ghz
    F, M = H.shape
    poles = initial_poles(f_ghz, n) if poles is None else poles
    for _ in range(iters):
        Phi = _basis(s, poles)
        N = Phi.shape[1]
        A = np.zeros((2 * F * M, M * (N + 1) + N))
        rhs = np.zeros(2 * F * M)
        for m in range(M):
            block = np.zeros((F, M * (N + 1) + N), dtype=complex)
            block[:, m * (N + 1):m * (N + 1) + N] = Phi
            block[:, m * (N + 1) + N] = 1
            block[:, M * (N + 1):] = -H[:, m:m + 1] * Phi
            A[2 * F * m:2 * F * (m + 1)], rhs[2 * F * m:2 * F * (m + 1)] = _realify(block, H[:, m])
        x, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        ct = x[M * (N + 1):]
        Am, b, k = np.zeros((N, N)), np.zeros(N), 0
        while k < N:
            a = poles[k]
            if a.imag != 0:
                Am[k:k + 2, k:k + 2] = [[a.real, a.imag], [-a.imag, a.real]]
                b[k] = 2
                k += 2
            else:
                Am[k, k], b[k] = a.real, 1
                k += 1
        poles = _arrange(np.linalg.eigvals(Am - np.outer(b, ct)))
    # final residue identification with fixed poles
    Phi = np.c_[_basis(s, poles), np.ones(F)]
    residues, consts, fits = [], [], []
    for m in range(M):
        Ar, br = _realify(Phi, H[:, m])
        c, *_ = np.linalg.lstsq(Ar, br, rcond=None)
        fits.append(Phi @ c)
        consts.append(c[-1])
        r, k = [], 0
        while k < len(poles):
            if poles[k].imag != 0:  # c1*phi1 + c2*phi2 -> residue (c1 + j c2) at a, conjugate at conj(a)
                z = c[k] + 1j * c[k + 1]
                r += [z, z.conjugate()]
                k += 2
            else:
                r.append(complex(c[k]))
                k += 1
        residues.append(r)
    return poles, np.array(residues), np.array(consts), np.stack(fits, 1)


def evaluate(f_ghz, poles, residues, consts):
    """Rebuild H (F, M) from a pole-residue model."""
    s = 2j * np.pi * f_ghz
    return consts[None, :] + (1 / (s[:, None] - poles[None, :])) @ residues.T


def natural_frequency_q(poles):
    """Natural frequency |p|/2pi (GHz) and Q = |p| / (2 |Re p|) of each pole."""
    return np.abs(poles) / (2 * np.pi), np.abs(poles) / (2 * np.abs(poles.real))
