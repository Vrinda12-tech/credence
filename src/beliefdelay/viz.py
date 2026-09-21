"""Draw the exact Bayes belief MAP next to the map decoded (linear probe) from each network's hidden state.

    python -m beliefdelay.viz results/cpu --env grid_patrol --archs lstm transformer rwkv mamba --out figures/maps.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from .metrics import collect, fit_probe
from .models import SeqModel
from .pomdp import make_env


def load_model(dirpath, env, arch, P, seed):
    stem = f"{env.name}__{arch}__P{P}__s{seed}"
    meta = json.load(open(os.path.join(dirpath, stem + ".json")))
    m = SeqModel(arch, env.token_fields, env.n_out, meta["d_model"], 2, max_len=meta["cfg"]["L"])
    m.load_state_dict(torch.load(os.path.join(dirpath, "ckpt", stem + ".pt"), map_location="cpu"))
    return m.eval(), meta["cfg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir"); ap.add_argument("--env", default="grid_patrol")
    ap.add_argument("--archs", nargs="+", default=["lstm", "transformer", "rwkv", "mamba"])
    ap.add_argument("--P", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--times", nargs="+", type=int, default=[2, 8, 20, 40])
    ap.add_argument("--traj", type=int, default=0, help="which held-out trajectory to draw")
    ap.add_argument("--out", default="figures/maps.png")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = make_env(a.env)
    if a.P is None:
        a.P = int(sorted(glob.glob(os.path.join(a.results_dir, f"{a.env}__*__P*__s{a.seed}.json")))[0].split("__P")[1].split("__")[0])
    H, W = env.H, env.W
    fig, axs = plt.subplots(1 + len(a.archs), len(a.times), figsize=(2.3 * len(a.times), 2.3 * (1 + len(a.archs))), squeeze=False)
    wall = ~env.free
    rows = [("Bayes (exact)", None)] + [(x, x) for x in a.archs]
    for ri, (label, arch) in enumerate(rows):
        if arch is None:
            model, cfg = None, None
        else:
            model, cfg = load_model(a.results_dir, env, arch, a.P, a.seed)
        L = cfg["L"] if cfg else 64
        d = collect(model if model else _Dummy(env), env, 768, L, seed=99)
        burn = cfg["burn_in"] if cfg else 24
        if model is not None:
            ntr = 512
            Hd, Y = d["hidden"][:, burn:], d["post"][:, burn:]
            Xtr, Ytr = Hd[:ntr].reshape(-1, Hd.shape[-1]), Y[:ntr].reshape(-1, Y.shape[-1])
            r = fit_probe(Xtr, Ytr, Hd[ntr:].reshape(-1, Hd.shape[-1])[:10], Y[ntr:].reshape(-1, Y.shape[-1])[:10])
        i = 512 + a.traj
        pos, obj = env.pos_of(d["obs"])[i], None
        states = None
        for ci, t in enumerate(a.times):
            t = min(t, L - 2)
            ax = axs[ri][ci]
            bayes = d["post"][i, t].numpy()
            m = bayes if arch is None else r["predict"](d["hidden"][i, t][None])[0].numpy()
            img = np.ma.array(m.reshape(H, W), mask=wall)
            ax.imshow(img, cmap="magma", vmin=0, vmax=max(0.5, float(bayes.max())))
            ax.imshow(np.where(wall, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=1)
            pr, pc = divmod(int(pos[t]), W)
            ax.plot(pc, pr, "s", ms=7, mfc="none", mec="cyan", mew=1.8)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(f"t={t}", fontsize=9)
            if ci == 0: ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f"{a.env}: belief map over the hidden object (cyan square = agent)", fontsize=10)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


class _Dummy(torch.nn.Module):
    """Placeholder 'model' so collect() can be reused for the Bayes row."""
    def __init__(self, env):
        super().__init__(); self.n = env.n_out; self.p = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.zeros(*x.shape, self.n), torch.zeros(*x.shape, 1)


if __name__ == "__main__":
    main()
