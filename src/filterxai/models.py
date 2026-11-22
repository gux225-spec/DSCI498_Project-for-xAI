"""Surrogate models (algorithm_plan_v2.md §2.3).

PRNet  9 inputs -> 5 shared conjugate pole pairs + residues -> S21(f), S11(f) complex
M0     9 inputs -> 601 S21_dB + 601 S11_dB (Suslova et al. 2026 layout)
M1     [9 inputs, frequency encoding] -> Re/Im S21, S11 at that frequency
All complex outputs are returned as (..., 2) complex tensors ordered [S21, S11].
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _inv_softplus(y):
    y = np.asarray(y, dtype=np.float64)
    return y + np.log(-np.expm1(-y))


def mlp(n_in, widths, act):
    layers, prev = [], n_in
    for w in widths:
        layers += [nn.Linear(prev, w), act()]
        prev = w
    return nn.Sequential(*layers), prev


class PRNet(nn.Module):
    """Pole-residue network. Poles ordered by natural frequency; stable by construction.

    f_k = f_min + cumsum(softplus(u))_k  (GHz),  Q_k = Q_min + softplus(v_k)
    p_k = 2*pi*f_k * (-1/(2Q_k) + j*sqrt(1 - 1/(4Q_k^2)))
    residue r_ck = (a_ck + j b_ck) * |Re p_k|   (scale-free: a+jb ~ peak contribution)
    S_c(f) = d_c + sum_k r_ck/(s - p_k) + conj(r_ck)/(s - conj(p_k)),  s = j*2*pi*f
    """

    def __init__(self, f_ghz, init, n_in=9, widths=(128, 128, 128), n_pairs=5, f_min=8.0, q_min=0.5, head_scale=0.1):
        super().__init__()
        self.K, self.f_min, self.q_min = n_pairs, f_min, q_min
        self.encoder, h = mlp(n_in, widths, nn.SiLU)
        self.head = nn.Linear(h, 2 * n_pairs + 4 * n_pairs + 2)
        self.register_buffer("s", torch.tensor(2j * np.pi * f_ghz, dtype=torch.complex64))
        with torch.no_grad():
            self.head.weight.mul_(head_scale)
            fn = np.asarray(init["natural_frequency_GHz"])
            u = _inv_softplus(np.diff(np.concatenate([[f_min], fn])))
            v = _inv_softplus(np.asarray(init["Q"]) - q_min)
            r = np.asarray(init["residue_normalized"])  # (2, K) complex
            bias = np.concatenate([u, v, r.real.ravel(), r.imag.ravel(), np.asarray(init["constant"])])
            self.head.bias.copy_(torch.tensor(bias, dtype=torch.float32))

    def concepts(self, x):
        """Natural frequency (GHz), Q, normalized residues (B, 2, K) complex, constants (B, 2)."""
        o = self.head(self.encoder(x))
        K = self.K
        fn = self.f_min + torch.cumsum(F.softplus(o[:, :K]), dim=1)
        q = self.q_min + F.softplus(o[:, K:2 * K])
        a = o[:, 2 * K:4 * K].reshape(-1, 2, K)
        b = o[:, 4 * K:6 * K].reshape(-1, 2, K)
        return fn, q, torch.complex(a, b), o[:, 6 * K:]

    def poles(self, fn, q):
        w = 2 * math.pi * fn
        re = -w / (2 * q)
        im = w * torch.sqrt(torch.clamp(1 - 1 / (4 * q * q), min=0.0))
        return torch.complex(re, im)

    def forward(self, x):
        return self.response(*self.concepts(x))

    def response(self, fn, q, rn, d):
        """S (B, F, 2) from concepts; lets concept-level games mix concepts across designs."""
        p = self.poles(fn, q)  # (B, K)
        r = rn * (-p.real)[:, None, :]  # (B, 2, K)
        s = self.s[None, :, None]  # (1, F, 1)
        g = 1 / (s - p[:, None, :])  # (B, F, K)
        gc = 1 / (s - p.conj()[:, None, :])
        S = torch.einsum("bfk,bck->bfc", g, r) + torch.einsum("bfk,bck->bfc", gc, r.conj())
        return S + d[:, None, :].to(S.dtype)


class M0(nn.Module):
    """Paper layout: 4 hidden x 256 with BatchNorm + LeakyReLU(0.1), linear output (5 layers)."""

    def __init__(self, n_in=9, n_out=1202, widths=(256, 256, 256, 256)):
        super().__init__()
        layers, prev = [], n_in
        for w in widths:
            layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.LeakyReLU(0.1)]
            prev = w
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def frequency_encoding(f_ghz, k_max):
    """u=(f-9)/6; k_max=0 -> [u]; else [u, sin(2 pi k u), cos(2 pi k u)] for k in 1,2,4,..,k_max."""
    u = (np.asarray(f_ghz) - 9.0) / 6.0
    cols = [u]
    k = 1
    while k_max and k <= k_max:
        cols += [np.sin(2 * np.pi * k * u), np.cos(2 * np.pi * k * u)]
        k *= 2
    return np.stack(cols, 1).astype(np.float32)


class M1(nn.Module):
    def __init__(self, n_in=9, n_freq=1, widths=(128, 128, 128)):
        super().__init__()
        self.body, h = mlp(n_in + n_freq, widths, nn.SiLU)
        self.out = nn.Linear(h, 4)

    def forward(self, x, fenc):
        o = self.out(self.body(torch.cat([x, fenc], dim=-1)))
        return torch.complex(o[..., 0::2], o[..., 1::2])  # [Re S21, Im S21, Re S11, Im S11] -> (.., 2)

    def curves(self, x, fenc):
        """Full curves for designs x (B, 9) on grid fenc (F, E) -> (B, F, 2)."""
        B, Fn = x.shape[0], fenc.shape[0]
        xx = x[:, None, :].expand(B, Fn, x.shape[1])
        ff = fenc[None].expand(B, Fn, fenc.shape[1])
        return self(xx, ff)


def prnet_init_from_vf(vf_npz):
    """Train-only medians of vector-fitting concepts, in PRNet's parameterization."""
    z = np.load(vf_npz)
    poles, res = z["poles"], z["residues"]
    up_idx = [np.flatnonzero(p.imag > 0)[np.argsort(np.abs(p[p.imag > 0]))] for p in poles]
    pu = np.stack([p[i] for p, i in zip(poles, up_idx)])  # (N, K)
    ru = np.stack([r[:, i] for r, i in zip(res, up_idx)])  # (N, 2, K)
    rn = ru / np.abs(pu.real)[:, None, :]
    return {
        "natural_frequency_GHz": np.median(z["natural_frequency_GHz"], 0),
        "Q": np.median(z["Q"], 0),
        "residue_normalized": np.median(rn.real, 0) + 1j * np.median(rn.imag, 0),
        "constant": np.median(z["constants"], 0),
    }
