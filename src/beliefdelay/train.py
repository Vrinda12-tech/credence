"""Online training on fresh POMDP samples (infinite data => no overfitting confound).

Loss = next-observation cross-entropy on SAMPLED observations (a proper scoring rule, so the
population minimiser is exactly the Bayes predictive law p*; the Bayes filter is never shown to the
model).  E[CE] = H* + E[KL(p*||p_theta)], hence  excess KL = CE - H*  is the training-objective gap.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .metrics import _dev, collect, excess_kl
from .pomdp import POMDP


LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2]   # starting grid; tuning.py extends it whenever the best value sits on an edge

PRESETS = {
    # smoke: seconds, only to check plumbing
    "smoke": dict(steps=60, batch=16, L=32, burn_in=16, kmax=8, eval_seqs=64, influence_seqs=64, tune_seqs=64,
                  lr=3e-3, target_params=30_000, seeds=[0], windows=[1, 2, 4, 8], lr_grid=[3e-3, 1e-2]),
    # cpu_quick: the settings of the first CPU run (1500 steps, 3 seeds): EXPLORATORY ONLY (3 seeds cannot reach alpha=0.05)
    "cpu_quick": dict(steps=1500, batch=32, L=48, burn_in=20, kmax=14, eval_seqs=256, influence_seqs=256, tune_seqs=512,
                      lr=3e-3, target_params=60_000, seeds=[0, 1, 2], windows=[1, 2, 3, 4, 6, 8, 12, 16, 24], lr_grid=LR_GRID),
    # cpu: laptop preset, 5 seeds (minimum for a test that can pass), longer training; expect many hours (Mamba dominates)
    "cpu": dict(steps=3000, batch=32, L=48, burn_in=20, kmax=14, eval_seqs=512, influence_seqs=512, tune_seqs=512,
                lr=3e-3, target_params=60_000, seeds=[0, 1, 2, 3, 4], windows=[1, 2, 3, 4, 6, 8, 12, 16, 24], lr_grid=LR_GRID),
    # small: GPU (Colab/Kaggle T4)
    "small": dict(steps=4000, batch=64, L=64, burn_in=24, kmax=16, eval_seqs=512, influence_seqs=512, tune_seqs=1024,
                  lr=2e-3, target_params=100_000, seeds=[0, 1, 2, 3, 4], windows=[1, 2, 3, 4, 6, 8, 12, 16, 24], lr_grid=LR_GRID),
    # full: the pre-registered configuration
    "full": dict(steps=10000, batch=64, L=64, burn_in=24, kmax=16, eval_seqs=1024, influence_seqs=1024, tune_seqs=1024,
                 lr=2e-3, target_params=100_000, seeds=[0, 1, 2, 3, 4], windows=[1, 2, 3, 4, 6, 8, 12, 16, 24], lr_grid=LR_GRID),
}


def train(model, env: POMDP, cfg: dict, seed: int, log_every: int = 250, verbose: bool = True):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    if "lr_core" in cfg and hasattr(model, "core_parameters"):
        opt = torch.optim.AdamW(
            [dict(params=model.core_parameters(), lr=cfg["lr_core"]), dict(params=model.head_parameters(), lr=cfg["lr"])],
            weight_decay=0.01, betas=(0.9, 0.98))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.98))
    steps, warm = cfg["steps"], max(1, cfg["steps"] // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    hist, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        obs, act, _ = env.sample(cfg["batch"], cfg["L"], gen)
        dv = _dev(model)
        logits, _ = model(env.tokens(obs, act).to(dv))
        loss = F.cross_entropy(logits.reshape(-1, env.n_out), env.targets(obs).reshape(-1).to(dv))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            d = collect(model, env, 128, cfg["L"], seed=777)
            ex = excess_kl(d, cfg["burn_in"])["excess_kl"]
            hist.append(dict(step=step, train_ce=float(loss.detach()), excess_kl=ex, sec=time.time() - t0))
            if verbose:
                print(f"    step {step:5d}  train_ce {float(loss.detach()):.4f}  excess_kl {ex:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return hist
