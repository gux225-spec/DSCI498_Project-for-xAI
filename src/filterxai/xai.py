"""Explainers and faithfulness tools (algorithm_plan_v2.md §4, §5A). Own implementations; shap/lime only
for the paper-protocol replication.

f: callable numpy (N, 9) -> (N, T). Attributions are returned as (D, T, 9).
"""
from math import factorial

import numpy as np
import torch
from scipy.stats import spearmanr

N = 9
MASKS = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(bool)  # coalition c contains i iff bit i


def shapley_matrix(n=N):
    """W (n, 2^n): phi_i = sum_c W[i, c] v(c)."""
    m = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    size = m.sum(1)
    W = np.zeros((n, 2 ** n))
    for i in range(n):
        without = ~m[:, i]
        w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in size[without]])
        c = np.flatnonzero(without)
        W[i, c] -= w
        W[i, c | (1 << i)] += w
    return W


W_SHAP = shapley_matrix()


def coalition_values(f, x, background):
    """v (D, 2^9, T): interventional game, features in the coalition from x, others from each background row."""
    out = []
    for xd in x:
        z = np.where(MASKS[:, None, :], xd[None, None, :], background[None, :, :])  # (512, B, 9)
        out.append(f(z.reshape(-1, N)).reshape(2 ** N, len(background), -1).mean(1))
    return np.stack(out)


def shap_from_values(v):
    return np.einsum("ic,dct->dti", W_SHAP, v)


def integrated_gradients(model, x, ref, steps=64, device="cpu", chunk=4):
    """IG with the trapezoid rule along the straight path ref -> x. Returns (D, T, 9) and F(x) - F(ref) (D, T)."""
    a = torch.linspace(0, 1, steps + 1, device=device)
    w = torch.full((steps + 1,), 1.0 / steps, device=device)
    w[[0, -1]] /= 2
    ref_t = torch.tensor(ref, device=device)
    phis, deltas = [], []
    for i in range(0, len(x), chunk):
        xc = torch.tensor(x[i:i + chunk], device=device)
        path = (ref_t + a[None, :, None] * (xc[:, None, :] - ref_t)).reshape(-1, N).requires_grad_(True)
        out = model(path)  # (C*(steps+1), T)
        g = torch.stack([torch.autograd.grad(out[:, t].sum(), path, retain_graph=t < out.shape[1] - 1)[0]
                         for t in range(out.shape[1])], 1)  # (P, T, 9)
        g = g.reshape(len(xc), steps + 1, out.shape[1], N)
        phis.append(((g * w[None, :, None, None]).sum(1) * (xc - ref_t)[:, None, :]).detach().cpu().numpy())
        o = out.detach().reshape(len(xc), steps + 1, -1)
        deltas.append((o[:, -1] - o[:, 0]).cpu().numpy())
    return np.concatenate(phis), np.concatenate(deltas)


def local_gradient(model, x, device="cpu"):
    """dF/dx at x: (D, T, 9)."""
    xt = torch.tensor(x, device=device).requires_grad_(True)
    out = model(xt)
    g = torch.stack([torch.autograd.grad(out[:, t].sum(), xt, retain_graph=t < out.shape[1] - 1)[0]
                     for t in range(out.shape[1])], 1)
    return g.detach().cpu().numpy()


def _truncated_normal(rng, x, scale, lo, hi, n):
    z = x + scale * rng.standard_normal((n, N))
    for _ in range(20):
        bad = (z < lo) | (z > hi)
        if not bad.any():
            break
        z[bad] = (x + scale * rng.standard_normal((n, N)))[bad]
    return np.clip(z, lo, hi)


def lime(f, x, lo, hi, scale=0.25, n_fit=1000, n_hold=500, seed=0, alpha=1e-6):
    """Bounded-neighbourhood LIME without discretization.

    Neighbours: Gaussian with std `scale` (train-std units) around x, truncated to the declared bounds.
    Kernel: exp(-|z - x|^2 / kw^2), kw = 0.75 * sqrt(9) * scale (LIME's default width in sampled units).
    Returns slopes (D, T, 9) per normalized unit and weighted held-out R2 (D, T).
    """
    rng = np.random.default_rng(seed)
    kw2 = (0.75 * np.sqrt(N) * scale) ** 2
    coefs, r2s = [], []
    for xd in x:
        z = _truncated_normal(rng, xd, scale, lo, hi, n_fit + n_hold)
        y = f(z)
        e = z - xd
        pi = np.exp(-(e ** 2).sum(1) / kw2)
        A = np.c_[np.ones(len(z)), e]
        Af, yf, pf = A[:n_fit], y[:n_fit], pi[:n_fit]
        M = Af.T @ (Af * pf[:, None]) + alpha * np.diag(np.r_[0, np.ones(N)])
        beta = np.linalg.solve(M, Af.T @ (yf * pf[:, None]))  # (10, T)
        Ah, yh, ph = A[n_fit:], y[n_fit:], pi[n_fit:]
        res = yh - Ah @ beta
        mu = (ph[:, None] * yh).sum(0) / ph.sum()
        r2s.append(1 - (ph[:, None] * res ** 2).sum(0) / np.maximum((ph[:, None] * (yh - mu) ** 2).sum(0), 1e-30))
        coefs.append(beta[1:].T)
    return np.stack(coefs), np.stack(r2s)


def deletion_insertion(v, phi, n_random=100, seed=0):
    """Deletion / insertion AUCs on the single-reference coalition table, and percentile vs random orders.

    Order: features by phi * sign(v(N) - v(empty)) descending (most gap-explaining first). Curves are
    normalized by the gap; deletion removes features from x (lower AUC better), insertion adds them to the
    reference (higher AUC better). Percentile = fraction of random orders at least as good (0 = best).
    v: (D, 512, T); phi: (D, T, 9). Returns dict of (D, T) arrays.
    """
    full = (1 << N) - 1
    gap = v[:, full] - v[:, 0]
    rng = np.random.default_rng(seed)
    rand = np.stack([rng.permutation(N) for _ in range(n_random)])

    def aucs(order):  # order (..., 9) -> deletion, insertion AUC for each (d, t)
        bits = np.cumsum(1 << order, axis=-1)  # coalition after inserting first k features
        ins_idx = np.concatenate([np.zeros(bits.shape[:-1] + (1,), int), bits], -1)  # k = 0..9
        return ins_idx, full - ins_idx

    D, _, T = v.shape
    out = {k: np.full((D, T), np.nan) for k in ("del_auc", "ins_auc", "del_pct", "ins_pct")}
    for d in range(D):
        for t in range(T):
            if abs(gap[d, t]) < 1e-12:
                continue
            curve = lambda idx: (v[d, idx, t] - v[d, 0, t]) / gap[d, t]  # noqa: E731
            ins_i, del_i = aucs(np.argsort(-phi[d, t] * np.sign(gap[d, t])))
            r_ins, r_del = aucs(rand)
            auc = lambda c: np.trapezoid(c, dx=1.0 / N, axis=-1)  # noqa: E731
            out["ins_auc"][d, t], out["del_auc"][d, t] = auc(curve(ins_i)), auc(curve(del_i))
            ri, rd = auc(curve(r_ins)), auc(curve(r_del))
            out["ins_pct"][d, t] = np.mean(ri >= out["ins_auc"][d, t])
            out["del_pct"][d, t] = np.mean(rd <= out["del_auc"][d, t])
    return out


def rank_corr(a, b):
    """Spearman over the last axis of |a| vs |b| (..., 9) -> (...); nan where either is constant."""
    a, b = np.abs(np.asarray(a)), np.abs(np.asarray(b))
    flat_a, flat_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    r = np.array([spearmanr(u, w)[0] if np.ptp(u) > 0 and np.ptp(w) > 0 else np.nan for u, w in zip(flat_a, flat_b)])
    return r.reshape(a.shape[:-1])


def intervention_sensitivity(f, x, background):
    """I (D, T, 9): mean over background rows of |F(x with x_j <- bg_j) - F(x)|."""
    fx = f(x)
    out = np.zeros((len(x),) + fx.shape[1:] + (N,))
    for j in range(N):
        z = np.repeat(x[:, None, :], len(background), 1)
        z[:, :, j] = background[None, :, j]
        out[..., j] = np.abs(f(z.reshape(-1, N)).reshape(len(x), len(background), -1) - fx[:, None]).mean(1)
    return out
