"""Kalman-oscillator experiment (Prop 7 in its purest form).

    python -m beliefdelay.run_lgssm --preset cpu --out results/lgssm
    python -m beliefdelay.run_lgssm --analyze results/lgssm --fig figures/kernels.png

Per (env, arch, seed): adaptive LR search on held-out MSE -> train on MSE of the next observation -> evaluate
  excess_mse  = E (mu*_t - f_t)^2        (exact; 0 iff Kalman-optimal)           skill = 1 - excess / E mu*^2
  state_r2    = held-out R^2 of a linear read-out of the Kalman state mean m_t from the last hidden layer
  kernel      = signed memory kernel g_theta(k) (perturb y_{t-k}), compared with the exact Kalman kernel:
                kernel_rel_err = ||g_theta - g*||_2 / ||g*||_2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .lgssm import LG_ENVS, make_lgssm
from .metrics import _dev, fit_ridge_r2
from .koopman import fit_dmdc, spectral_match
from .models import ARCHS, EXTRA_ARCHS, SeqModel, build_matched
from .tuning import adaptive_lr_search

LG_PRESETS = {
    "smoke": dict(steps=80, batch=32, L=32, burn_in=12, kmax=10, eval_seqs=128, kernel_seqs=64, tune_seqs=128,
                  lr=3e-3, target_params=20_000, seeds=[0], lr_grid=[3e-3, 1e-2]),
    "cpu": dict(steps=3000, batch=64, L=48, burn_in=16, kmax=16, eval_seqs=512, kernel_seqs=256, tune_seqs=512,
                lr=3e-3, target_params=40_000, seeds=[0, 1, 2, 3, 4], lr_grid=[1e-3, 3e-3, 1e-2, 3e-2]),
    "full": dict(steps=8000, batch=64, L=64, burn_in=16, kmax=20, eval_seqs=1024, kernel_seqs=512, tune_seqs=1024,
                 lr=3e-3, target_params=40_000, seeds=[0, 1, 2, 3, 4], lr_grid=[1e-3, 3e-3, 1e-2, 3e-2]),
}


def train_lg(model, env, cfg, seed, lr, steps=None, log_every=500, verbose=True):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    steps = steps or cfg["steps"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(1, steps // 20)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    dv, hist, t0 = _dev(model), [], time.time()
    model.train()
    for step in range(1, steps + 1):
        y, _ = env.sample(cfg["batch"], cfg["L"], gen)
        out, _ = model(y[:, :-1, None].float().to(dv))
        loss = F.mse_loss(out[..., 0], y[:, 1:].float().to(dv))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps:
            ex = excess_mse(model, env, 256, cfg["L"], cfg["burn_in"], seed=777)["excess_mse"]
            hist.append(dict(step=step, train_mse=float(loss.detach()), excess_mse=ex, sec=time.time() - t0))
            if verbose:
                print(f"    step {step:5d}  train_mse {float(loss.detach()):.4f}  excess_mse {ex:.5f}", flush=True)
    return hist


@torch.no_grad()
def _run(model, y):
    was = model.training
    model.eval()
    out, h = model(y[:, :-1, None].float().to(_dev(model)))
    model.train(was)
    return out[..., 0].double().cpu(), h.float().cpu()


def excess_mse(model, env, n, L, burn, seed):
    y, _ = env.sample(n, L, torch.Generator().manual_seed(seed))
    _, mu, S = env.kalman(y)
    f, _ = _run(model, y)
    ex = ((f - mu)[:, burn:] ** 2).mean()
    power = (mu[:, burn:] ** 2).mean()
    return dict(excess_mse=float(ex), signal_power=float(power), skill=float(1 - ex / power),
                bayes_mse=float(S[burn:].mean()), mse=float(((f - y[:, 1:])[:, burn:] ** 2).mean()))


@torch.no_grad()
def model_kernel(model, env, kmax, n, L, seed, eps=0.5):
    """Signed empirical kernel g_theta(k) = E[ (f_t(y + eps e_{t-k}) - f_t(y)) / eps ]."""
    gen = torch.Generator().manual_seed(seed)
    y, _ = env.sample(n, L, gen)
    t = L - 1
    base = _run(model, y)[0][:, t]
    g = []
    for k in range(kmax + 1):
        y2 = y.clone()
        y2[:, t - k] += eps
        g.append(float(((_run(model, y2)[0][:, t] - base) / eps).mean()))
    return np.array(g)


def evaluate(model, env, cfg, seed=12345):
    res = excess_mse(model, env, cfg["eval_seqs"], cfg["L"], cfg["burn_in"], seed)
    n, L, burn = cfg["eval_seqs"], cfg["L"], cfg["burn_in"]
    y, _ = env.sample(n, L, torch.Generator().manual_seed(seed))
    m_post, _, _ = env.kalman(y)
    _, h = _run(model, y)
    ntr = int(0.7 * n)
    Hh, M = h[:, burn:], m_post[:, burn:].float()
    r2 = fit_ridge_r2(Hh[:ntr].reshape(-1, Hh.shape[-1]), M[:ntr].reshape(-1, 2), Hh[ntr:].reshape(-1, Hh.shape[-1]), M[ntr:].reshape(-1, 2))
    res["state_r2"] = float(r2.mean())
    g_star = env.kernel(cfg["kmax"], L=max(L, cfg["kmax"] + 20))
    g = model_kernel(model, env, cfg["kmax"], cfg["kernel_seqs"], L, seed + 1)
    res["kernel_model"], res["kernel_kalman"] = g.tolist(), g_star.tolist()
    res["kernel_rel_err"] = float(np.linalg.norm(g - g_star) / max(np.linalg.norm(g_star), 1e-12))
    return res


def tune(arch, env, cfg, P, device):
    def score(lr):
        torch.manual_seed(999)
        m = build_matched(arch, None, 1, P, max_len=cfg["L"], in_dim=1).to(device)
        train_lg(m, env, cfg, 999, lr, steps=max(50, cfg["steps"] // 2), verbose=False, log_every=10 ** 9)
        y, _ = env.sample(cfg["tune_seqs"], cfg["L"], torch.Generator().manual_seed(31337))
        return float(((_run(m, y)[0] - y[:, 1:])[:, cfg["burn_in"]:] ** 2).mean())
    return adaptive_lr_search(score, cfg["lr_grid"], default=cfg["lr"], tol=2e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=list(LG_PRESETS))
    ap.add_argument("--envs", nargs="+", default=LG_ENVS)
    ap.add_argument("--archs", nargs="+", default=ARCHS + ["delay_mlp"], help=f"extras: {EXTRA_ARCHS}")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--params", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="results/lgssm")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--L", type=int, default=None, help="sequence length override (use with the *_slow envs)")
    ap.add_argument("--kmax", type=int, default=None, help="kernel length override")
    ap.add_argument("--spectrum", default=None, help="results dir: Koopman/DMDc spectrum of hidden dynamics per (env, arch); needs checkpoints")
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--fig", default="figures/kernels.png")
    a = ap.parse_args()
    if a.analyze:
        return analyze(a.analyze, a.fig)
    if a.spectrum:
        return spectrum(a.spectrum)
    cfg = dict(LG_PRESETS[a.preset])
    if a.L:
        cfg["L"] = a.L
    if a.kmax:
        cfg["kmax"] = a.kmax
    if a.steps:
        cfg["steps"] = a.steps
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    P = a.params or cfg["target_params"]
    if len(seeds) < 5:
        print(f"WARNING: {len(seeds)} seeds: exploratory only.", flush=True)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    os.makedirs(a.out, exist_ok=True)
    for env_name in a.envs:
        env = make_lgssm(env_name)
        for arch in a.archs:
            for seed in seeds:
                path = os.path.join(a.out, f"{env_name}__{arch}__P{P}__s{seed}.json")
                if os.path.exists(path):
                    continue
                print(f"[{env_name}] {arch} P={P} seed={seed}", flush=True)
                lrp = os.path.join(a.out, f"lr__{env_name}__{arch}__P{P}.json")
                if os.path.exists(lrp):
                    lr_info = json.load(open(lrp))
                else:
                    lr_info = tune(arch, env, cfg, P, device)
                    json.dump(lr_info, open(lrp, "w"))
                torch.manual_seed(seed)
                model = build_matched(arch, None, 1, P, max_len=cfg["L"], in_dim=1).to(device)
                hist = train_lg(model, env, cfg, seed, lr_info["best"])
                res = evaluate(model, env, cfg)
                res.update(env=env_name, arch=arch, seed=seed, n_params=model.n_params(), d_model=model.d, lr=lr_info["best"],
                           lr_at_bound=lr_info.get("at_bound"), history=hist, eig=env.meta["eig"],
                           cfg={k: v for k, v in cfg.items() if k != "seeds"})
                os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(a.out, "ckpt", os.path.basename(path).replace(".json", ".pt")))
                json.dump(res, open(path, "w"))
                print(f"    -> excess_mse {res['excess_mse']:.5f} skill {res['skill']:.3f} state_r2 {res['state_r2']:.3f} "
                      f"kernel_err {res['kernel_rel_err']:.3f}", flush=True)


def spectrum(dirpath, rank=4, n=512):
    """Koopman/DMDc spectrum of each trained model's last-layer hidden dynamics vs the exact closed-loop (filter) spectrum."""
    import pandas as pd
    from .koopman import fit_dmdc, spectral_match
    rows = []
    for f in sorted(glob.glob(os.path.join(dirpath, "*__s*.json"))):
        if "lr__" in f:
            continue
        r = json.load(open(f))
        ck = os.path.join(dirpath, "ckpt", os.path.basename(f).replace(".json", ".pt"))
        if not os.path.exists(ck):
            continue
        env = make_lgssm(r["env"])
        m = SeqModel(r["arch"], None, 1, r["d_model"], 2, max_len=r["cfg"]["L"], in_dim=1)
        m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
        y, _ = env.sample(n, r["cfg"]["L"], torch.Generator().manual_seed(2024))
        _, h = _run(m, y)
        U = y[:, :-1, None].float()
        fit = fit_dmdc(h, U, rank=rank, burn=r["cfg"]["burn_in"])
        sm = spectral_match(fit["eig"], env.closed_loop_eig())
        rows.append(dict(env=r["env"], arch=r["arch"], seed=r["seed"], fit_r2=fit["r2"], spec_dist=sm["spec_dist"],
                         has_rotation=sm["has_rotation"], lead_angle_deg=sm["lead_angle_deg"], lead_mod=abs(sm["lead_complex"])))
    if not rows:
        print("no checkpoints found (re-run without --analyze; checkpoints are saved to <out>/ckpt)")
        return
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for env in sorted(df.env.unique()):
        print(f"\n{env}: exact filter (Koopman-with-input) spectrum = {np.round(make_lgssm(env).closed_loop_eig(), 3)}"
              f"   [angle {np.degrees(np.abs(np.angle(make_lgssm(env).closed_loop_eig()))).round(1)} deg]")
        print(df[df.env == env].groupby("arch")[["fit_r2", "spec_dist", "has_rotation", "lead_angle_deg", "lead_mod"]].mean().round(3).to_string())
    print("\nreading guide: spec_dist ~ 0 and matching angle => hidden dynamics reproduce the true spectrum; low fit_r2 => linear read of the hidden state is unreliable; "
          "transformer rows are descriptive only (its state is not recurrent).")


def analyze(dirpath, fig):
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(dirpath, "*__s*.json"))) if "lr__" not in f]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.groupby(["env", "arch"])[["excess_mse", "skill", "state_r2", "kernel_rel_err"]].agg(["mean", "std"]).round(4).to_string())
    print("\nper-seed excess_mse:")
    print(df.pivot_table(index=["env", "arch"], columns="seed", values="excess_mse").round(5).to_string())
    print("\nlr at bound (untrustworthy if True):", df[df.get("lr_at_bound", False) == True][["env", "arch"]].drop_duplicates().values.tolist())
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    envs = sorted(df.env.unique())
    col = dict(lstm="#8c564b", transformer="#1f77b4", rwkv="#2ca02c", mamba="#d62728", delay_mlp="#7f7f7f")
    fig_, axs = plt.subplots(1, len(envs), figsize=(5 * len(envs), 3.6), squeeze=False)
    for ax, env in zip(axs[0], envs):
        g = df[df.env == env]
        ax.plot(np.mean(np.stack(g.kernel_kalman.values), 0), "k-o", lw=2.5, label="Kalman (exact)")
        for arch, ga in g.groupby("arch"):
            ax.plot(np.mean(np.stack(ga.kernel_model.values), 0), "-s", color=col.get(arch), label=arch, alpha=0.85)
        ax.axhline(0, color="gray", lw=0.5); ax.set_title(env); ax.set_xlabel("lag k")
    axs[0][0].set_ylabel("memory kernel g(k)"); axs[0][0].legend(fontsize=7)
    os.makedirs(os.path.dirname(fig) or ".", exist_ok=True)
    fig_.tight_layout(); fig_.savefig(fig, dpi=140)
    print("wrote", fig)


if __name__ == "__main__":
    main()
