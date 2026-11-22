"""Frozen step-1 models (results/step1/final_models.json) as target functions x (B, 9) -> F (B, T).

Every surrogate is wrapped the same way: inputs -> dB curves -> targets.CurveTargets, plus T3 concepts
for PR-Net. `torch_targets` keeps the graph (IG, GradientExplainer); `targets` evaluates in chunks without it.
"""
import json

import numpy as np
import torch
from torch import nn

from . import data, models, targets
from .train import DEV

RUNS = data.ROOT / "results" / "step1" / "runs"
REGISTRY = data.ROOT / "results" / "step1" / "final_models.json"
CHUNK_DESIGNS = {"PRNet": 4096, "M0": 8192, "M1": 512}


class Surrogate(nn.Module):
    def __init__(self, kind, tag):
        super().__init__()
        self.kind, self.tag = kind, tag
        tr = data.load("train")
        f = tr["frequency_GHz"]
        cfg = json.loads(REGISTRY.read_text(encoding="utf-8"))[kind]["config"]
        state = torch.load(RUNS / tag / "model.pt", map_location="cpu")
        if kind == "PRNet":
            self.net = models.PRNet(f, models.prnet_init_from_vf(data.ROOT / "results" / "step0" / "vf_concepts_train.npz"))
        elif kind == "M1":
            self.net = models.M1(n_freq=models.frequency_encoding(f, cfg["fourier_k"]).shape[1])
            self.register_buffer("fenc", torch.tensor(models.frequency_encoding(f, cfg["fourier_k"])))
        else:
            self.net = models.M0()
            y = tr["Y_dB"]  # same train-only z-score as train.train_m0
            self.register_buffer("mu", torch.tensor(y.mean(0).T.ravel(), dtype=torch.float32))
            self.register_buffer("sd", torch.tensor(y.std(0).T.ravel(), dtype=torch.float32))
        self.net.load_state_dict(state)
        self.T = targets.CurveTargets(f, DEV)
        self.names = targets.CURVE_TARGETS + (targets.CONCEPT_TARGETS if kind == "PRNet" else [])
        self.to(DEV).eval()

    def db(self, x):
        if self.kind == "M0":
            y = self.net(x) * self.sd + self.mu
            return y.reshape(len(x), 2, -1).transpose(1, 2)
        S = self.net(x) if self.kind == "PRNet" else self.net.curves(x, self.fenc)
        return 10 * torch.log10(torch.clamp(torch.abs(S) ** 2, min=1e-30))

    def forward(self, x):
        """Targets (B, T) with graph; T3 = resonator poles 2-4 natural frequency and Q for PR-Net."""
        out = self.T(self.db(x))
        if self.kind == "PRNet":
            fn, q, _, _ = self.net.concepts(x)
            out = torch.cat([out, fn[:, 1:4], q[:, 1:4]], dim=1)
        return out

    @torch.no_grad()
    def targets(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 9)
        n = CHUNK_DESIGNS[self.kind]
        return np.concatenate([self(torch.tensor(x[i:i + n], device=DEV)).cpu().numpy() for i in range(0, len(x), n)])


def load(kind, seed=None):
    """Surrogates of one kind from the registry (all seeds, or the one with `seed`)."""
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))[kind]["runs"]
    tags = [t for t in reg if seed is None or f"_s{seed}" in t]
    return [Surrogate(kind, t) for t in tags]
