"""Empirical reproduction of Luo et al. 2024 (NeurIPS, "Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific
Learning Rate", arXiv:2405.15384), Proposition 1, on OUR architectures.

Their claim (loss-agnostic; only assumes Lipschitz continuity of the network map): if the hidden recurrence is contractive,
i.e. there is K_h in [0,1) with  ||f_h(x,h) - f_h(x,h')|| <= K_h ||h-h'||  for every input x, then perturbing the parameters by
a fixed one-step gradient update of size eps changes the output at rollout position t by at most

    ||y_t - y_t'|| <= K_y (1-K_h^t)/(1-K_h) eps + eps        (Prop 1)

which GROWS with t but CONVERGES to K_y/(1-K_h) eps + eps.  GRU/LSTM (sigmoid gates) and Mamba/RWKV (bounded per-channel
decay, i.e. the real-diagonal mechanism of THEORY.md Prop 7/8) all satisfy K_h < 1; a Transformer's hidden dynamics is not a
fixed-point recurrence in time in the same sense, so the bound does not obviously apply to it (checked, not assumed, below).

`amplification_curve` measures ||y_t(theta) - y_t(theta+delta)|| for t = 0..T on FRESH random histories, for a perturbation
delta applied only to the model's core parameters (models.core_parameters()).  This is the exact experiment behind their
Figure "policy output variations as rollout step increases", reproduced on our sequence models and our synthetic data
instead of theirs, and used as a design check: if our recurrent architectures show the predicted amplify-then-plateau
shape, using one global learning rate for the whole network (as our first CPU run did) is confounded exactly as RESeL warns.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def amplification_curve(model, tokens: torch.Tensor, eps: float = 1e-3, seed: int = 0):
    """tokens: (B, T) fresh input sequences.  Returns ||y_t - y_t'||_2 averaged over the batch, for t=0..T-1,
    where theta' perturbs ONLY the core parameters by a random unit direction scaled to eps (a stand-in for
    "one gradient step of size eps"; the direction is what varies across parameter space, RESeL's bound holds
    for any such perturbation of bounded norm)."""
    base = model(tokens)[0].double()
    gen = torch.Generator().manual_seed(seed)
    core = model.core_parameters()
    saved = [p.detach().clone() for p in core]
    try:
        for p in core:
            d = torch.randn(p.shape, generator=gen, dtype=p.dtype)
            d = d / (d.norm() + 1e-12) * eps
            p.add_(d)
        out = model(tokens)[0].double()
    finally:
        for p, s in zip(core, saved):
            p.copy_(s)
    diff = (out - base).norm(dim=-1)          # (B, T)
    return diff.mean(0).numpy()


def theory_curve(K_h: float, K_y: float, eps: float, T: int) -> np.ndarray:
    """The exact RHS of Prop 1 as a function of t (an upper bound, not a prediction of the measured value)."""
    t = np.arange(T)
    amp = K_y * (1 - K_h ** t) / (1 - K_h) if K_h < 1 else K_y * t
    return amp * eps + eps


def fit_Kh(curve: np.ndarray) -> dict:
    """Fit the plateau shape c(t) = A(1 - r^t) + B by nonlinear least squares (r = fitted K_h, only meaningful if
    the curve actually plateaus, i.e. is non-decreasing and bounded)."""
    t = np.arange(len(curve), dtype=float)
    best = None
    for r in np.linspace(0.01, 0.995, 100):
        basis = np.stack([1 - r ** t, np.ones_like(t)], 1)
        coef, *_ = np.linalg.lstsq(basis, curve, rcond=None)
        resid = float(np.sum((basis @ coef - curve) ** 2))
        if best is None or resid < best[0]:
            best = (resid, r, coef)
    _, r, (A, B) = best
    plateau = curve[-max(3, len(curve) // 10):].mean()
    early = curve[:3].mean()
    monotone_frac = float(np.mean(np.diff(curve) >= -1e-9))
    return dict(fitted_Kh=float(r), A=float(A), B=float(B), plateau=float(plateau), early=float(early),
                growth_ratio=float(plateau / max(early, 1e-12)), monotone_frac=monotone_frac,
                looks_contractive=bool(monotone_frac > 0.8 and plateau < 1e6))
