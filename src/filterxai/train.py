"""Training loops for PRNet, M0, M1 (algorithm_plan_v2.md §2.4). Validation only; test is never loaded."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data, metrics, models

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Context:
    """Train/validation tensors and train-only statistics shared by all models."""

    def __init__(self, control_replicate=0, preset=data.PRESET):
        self.tr = data.load("train", control_replicate, preset)
        self.va = data.load("validation", control_replicate, preset)
        self.f = self.tr["frequency_GHz"]
        masks = data.region_masks(self.f)
        w = np.zeros(len(self.f))
        for m in masks.values():
            w[m] = 1.0 / (len(masks) * m.sum())  # sum_j w_j * l_j = mean over regions of region means
        self.w = torch.tensor(w, dtype=torch.float32, device=DEV)
        s21 = self.tr["Y_dB"][..., 0]
        self.s21_mu, self.s21_sd = float(s21.mean()), float(s21.std())
        self.ydb_mu = self.tr["Y_dB"].mean(0)  # (F, 2) for M0
        self.ydb_sd = self.tr["Y_dB"].std(0)
        t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEV)  # noqa: E731
        self.Xtr, self.Xva = t(self.tr["X"]), t(self.va["X"])
        self.Str, self.Sva = t(self.tr["S"], torch.complex64), t(self.va["S"], torch.complex64)


def rf_loss(ctx, S_pred, S_true, lam, w):
    """Region-weighted complex MSE + lam * region-weighted Huber on standardized S21_dB.

    S_*: (..., F', 2) complex; w: matching (F',) or broadcastable point weights summing to 1 per curve.
    """
    cm = (torch.abs(S_pred - S_true) ** 2).mean(-1)
    db_p = 10 * torch.log10(torch.abs(S_pred[..., 0]) ** 2 + 1e-12)
    db_t = 10 * torch.log10(torch.abs(S_true[..., 0]) ** 2 + 1e-12)
    hub = F.huber_loss((db_p - ctx.s21_mu) / ctx.s21_sd, (db_t - ctx.s21_mu) / ctx.s21_sd, reduction="none", delta=1.0)
    w = w.expand_as(cm)
    return ((cm + lam * hub) * w).sum() / w.sum()


def power_penalty(S_pred, w, mode):
    """Region-weighted soft constraint on P = |S21|^2 + |S11|^2 (port-1 power balance).

    mode "passive":  relu(P - 1)^2   (only P > 1 violates passivity)
    mode "lossless": (P - 1)^2       (the ADS data is lossless, P = 1 to 3e-15)
    """
    d = (torch.abs(S_pred) ** 2).sum(-1) - 1
    d = F.relu(d) if mode == "passive" else d
    w = w.expand_as(d)
    return (d ** 2 * w).sum() / w.sum()


def _loop(model, opt, epoch_fn, val_fn, max_epochs, patience, scheduler=None, log_every=100):
    best, best_state, wait, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        tl = epoch_fn()
        model.eval()
        with torch.no_grad():
            vl = val_fn()
        if scheduler is not None:
            scheduler.step(vl)
        hist.append((ep, tl, vl, opt.param_groups[0]["lr"]))
        if vl < best:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if ep % log_every == 0:
            print(f"  ep {ep:3d} train {tl:.3e} val {vl:.3e} best {best:.3e} ({time.time() - t0:.0f}s)", flush=True)
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"epochs_run": ep + 1, "best_val_loss": float(best), "seconds": round(time.time() - t0, 1),
            "history": [list(map(float, h)) for h in hist]}


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _plateau(opt, on):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10) if on else None


def train_prnet(ctx, init, lam, seed, max_epochs=300, patience=30, batch=64, lr=1e-3, wd=1e-4, plateau=True,
                mu=0.0, pen="lossless"):
    """mu > 0 adds mu * power_penalty to the training loss only; early stopping stays on rf_loss."""
    _seed(seed)
    model = models.PRNet(ctx.f, init).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _plateau(opt, plateau)
    n = len(ctx.Xtr)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def epoch():
        tot = 0.0
        for idx in torch.randperm(n, generator=g).split(batch):
            idx = idx.to(DEV)
            S = model(ctx.Xtr[idx])
            loss = rf_loss(ctx, S, ctx.Str[idx], lam, ctx.w)
            if mu:
                loss = loss + mu * power_penalty(S, ctx.w, pen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        return tot / n

    info = _loop(model, opt, epoch, lambda: rf_loss(ctx, model(ctx.Xva), ctx.Sva, lam, ctx.w).item(), max_epochs, patience, sched)
    return model, info


def train_m0(ctx, seed, max_epochs=300, patience=30, batch=64, lr=1e-3, wd=1e-4):
    _seed(seed)
    model = models.M0().to(DEV)
    mu = torch.tensor(ctx.ydb_mu.T.ravel(), dtype=torch.float32, device=DEV)  # [S21 (F), S11 (F)]
    sd = torch.tensor(ctx.ydb_sd.T.ravel(), dtype=torch.float32, device=DEV)
    z = lambda y: (torch.tensor(y.transpose(0, 2, 1).reshape(len(y), -1), dtype=torch.float32, device=DEV) - mu) / sd  # noqa: E731
    Ztr, Zva = z(ctx.tr["Y_dB"]), z(ctx.va["Y_dB"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _plateau(opt, True)
    n = len(ctx.Xtr)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def epoch():
        tot = 0.0
        for idx in torch.randperm(n, generator=g).split(batch):
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            idx = idx.to(DEV)
            loss = F.mse_loss(model(ctx.Xtr[idx]), Ztr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        return tot / n

    info = _loop(model, opt, epoch, lambda: F.mse_loss(model(ctx.Xva), Zva).item(), max_epochs, patience, sched)
    model.out_mu, model.out_sd = mu, sd
    return model, info


def train_m1(ctx, k_max, lam, seed, max_epochs=300, patience=30, batch=4096, lr=1e-3, wd=1e-4, plateau=True):
    _seed(seed)
    fenc = torch.tensor(models.frequency_encoding(ctx.f, k_max), device=DEV)
    model = models.M1(n_freq=fenc.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _plateau(opt, plateau)
    n, Fn = ctx.Xtr.shape[0], len(ctx.f)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def epoch():
        tot, cnt = 0.0, 0
        for rows in torch.randperm(n * Fn, generator=g).split(batch):
            rows = rows.to(DEV)
            i, j = rows // Fn, rows % Fn
            S = model(ctx.Xtr[i], fenc[j])
            loss = rf_loss(ctx, S, ctx.Str[i, j], lam, ctx.w[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(rows)
            cnt += len(rows)
        return tot / cnt

    info = _loop(model, opt, epoch, lambda: rf_loss(ctx, model.curves(ctx.Xva, fenc), ctx.Sva, lam, ctx.w).item(),
                 max_epochs, patience, sched, log_every=25)
    model.fenc = fenc
    return model, info


@torch.no_grad()
def predict(kind, model, X):
    """Validation predictions as numpy: (dB (N,F,2), complex S or None)."""
    model.eval()
    if kind == "PRNet":
        S = model(X).cpu().numpy()
        return metrics.to_db(S), S
    if kind == "M1":
        S = model.curves(X, model.fenc).cpu().numpy()
        return metrics.to_db(S), S
    if kind == "M0":
        y = (model(X) * model.out_sd + model.out_mu).cpu().numpy()
        return y.reshape(len(y), 2, -1).transpose(0, 2, 1), None
    raise ValueError(kind)
