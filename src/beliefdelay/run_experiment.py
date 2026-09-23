"""CLI:  python -m beliefdelay.run_experiment --preset small --out results/small

One JSON per (env, arch, params, seed).  Existing files are skipped, so a Colab/Kaggle disconnect
only costs the run in progress (point --out at Google Drive / a Kaggle output dir to persist).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .metrics import collect, evaluate_model, heldout_ce, window_error_curve
from .tuning import adaptive_lr_search, two_stage_lr_search
from .models import ARCHS, EXTRA_ARCHS, build_matched
from .grid import GRID_ENVS
from .pomdp import ENVS, make_env
from .train import PRESETS, train


def tune_lr(arch, env, cfg, P, grid, device, frac=0.5, split=False):
    """Held-out-CE (oracle-free) LR search at `frac` of training length; extends past grid edges, defaults on a flat grid.
    split=True: RESeL-motivated (Luo et al. 2024, arXiv:2405.15384) two-stage core/head search (see tuning.py)."""
    def make(seed=999):
        torch.manual_seed(seed)
        return build_matched(arch, env.token_fields, env.n_out, P, max_len=cfg["L"]).to(device)

    def score_joint(lr):
        c = dict(cfg, lr=lr, steps=max(50, int(cfg["steps"] * frac)))
        m = make()
        train(m, env, c, seed=999, log_every=10 ** 9, verbose=False)
        return heldout_ce(collect(m, env, cfg.get("tune_seqs", 512), cfg["L"], seed=31337), cfg["burn_in"])

    if not split:
        return adaptive_lr_search(score_joint, grid, default=cfg["lr"])

    def score_split(core_lr, head_lr):
        c = dict(cfg, lr=head_lr, lr_core=core_lr, steps=max(50, int(cfg["steps"] * frac)))
        m = make()
        train(m, env, c, seed=999, log_every=10 ** 9, verbose=False)
        return heldout_ce(collect(m, env, cfg.get("tune_seqs", 512), cfg["L"], seed=31337), cfg["burn_in"])

    return two_stage_lr_search(score_joint, score_split, grid, default=cfg["lr"])


def convergence_flag(hist, steps):
    """True if excess KL was still falling by >15% between 80% of training and the end (=> ranking mixes speed and asymptote)."""
    late = [h for h in hist if h["step"] >= 0.8 * steps]
    if len(late) < 2:
        return False
    return (late[0]["excess_kl"] - late[-1]["excess_kl"]) / max(late[0]["excess_kl"], 1e-12) > 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--envs", nargs="+", default=["grid_static", "grid_drift", "grid_patrol"],
                    help=f"grid: {list(GRID_ENVS)} | abstract sanity envs: {list(ENVS)}")
    ap.add_argument("--archs", nargs="+", default=ARCHS, help=f"default {ARCHS}; extras: {EXTRA_ARCHS}")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--params", nargs="+", type=int, default=None, help="parameter budgets (capacity sweep)")
    ap.add_argument("--lr-grid", nargs="+", type=float, default=None,
                    help="starting grid for the per-(env,arch) LR search (default: the preset's); extended automatically at edges")
    ap.add_argument("--no-tune", action="store_true", help="use the preset's single LR (NOT for comparisons between architectures)")
    ap.add_argument("--split-lr", action="store_true",
                    help="RESeL-motivated (Luo et al. 2024, arXiv:2405.15384): search a separate, smaller learning rate "
                         "for the recurrent core than for the embeddings/head. Exploratory; see PREREGISTRATION.md.")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.steps:
        cfg["steps"] = args.steps
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if len(seeds) < 5:
        print(f"WARNING: {len(seeds)} seeds. A permutation test with n<5 per arm cannot reach alpha=0.05 after Holm; "
              f"treat as exploratory.", flush=True)
    grid = args.lr_grid or cfg["lr_grid"]
    budgets = args.params or [cfg["target_params"]]
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    print(f"device: {device}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    for env_name in args.envs:
        env = make_env(env_name)
        eps_path = os.path.join(args.out, f"windowcurve_{env_name}.json")
        if os.path.exists(eps_path):
            eps_curve = {int(k): v for k, v in json.load(open(eps_path)).items()}
        else:
            print(f"[{env_name}] window-filter error curve ...", flush=True)
            eps_curve = window_error_curve(env, cfg["windows"], 64, cfg["L"], cfg["burn_in"], seed=4242)
            json.dump(eps_curve, open(eps_path, "w"))
        for P in budgets:
            for arch in args.archs:
                for seed in seeds:
                    path = os.path.join(args.out, f"{env_name}__{arch}__P{P}__s{seed}.json")
                    if os.path.exists(path):
                        continue
                    print(f"[{env_name}] {arch} P={P} seed={seed}", flush=True)
                    run_cfg = dict(cfg)
                    if not args.no_tune:
                        lrp = os.path.join(args.out, f"lr__{env_name}__{arch}__P{P}.json")
                        if os.path.exists(lrp):
                            lr_info = json.load(open(lrp))
                        else:
                            lr_info = tune_lr(arch, env, cfg, P, grid, device, split=args.split_lr)
                            json.dump(lr_info, open(lrp, "w"))
                        if args.split_lr:
                            run_cfg["lr"], run_cfg["lr_core"] = lr_info["head_lr"], lr_info["core_lr"]
                        else:
                            run_cfg["lr"] = lr_info["best"]
                    torch.manual_seed(seed)
                    model = build_matched(arch, env.token_fields, env.n_out, P, max_len=cfg["L"]).to(device)
                    t0 = time.time()
                    hist = train(model, env, run_cfg, seed)
                    res = evaluate_model(model, env, run_cfg, eps_curve)
                    res.update(converged=not convergence_flag(hist, run_cfg["steps"]),
                               lr_at_bound=(lr_info.get("at_bound", lr_info.get("ratio_search", {}).get("at_bound")) if not args.no_tune else None),
                               lr_flat=(lr_info.get("flat", lr_info.get("ratio_search", {}).get("flat")) if not args.no_tune else None),
                               split_lr=args.split_lr, env=env_name, arch=arch, seed=seed, target_params=P, n_params=model.n_params(),
                               d_model=model.d, train_sec=time.time() - t0, history=hist,
                               bayes_dobrushin=env.dobrushin(), lr=run_cfg["lr"], cfg={k: v for k, v in run_cfg.items() if k != "seeds"})
                    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
                    torch.save(model.state_dict(), path.replace(args.out, os.path.join(args.out, "ckpt")).replace(".json", ".pt"))
                    json.dump(res, open(path, "w"))
                    print(f"    -> excess_kl {res['excess_kl']:.5f}  probe_pred {res['probe_pred_score']:.3f}  "
                          f"psr {res['probe_psr_score']:.3f}  belief {res['probe_belief_score']:.3f}  skill {res['skill']:.3f}  W1 {res['prof_w1']:.2f}  L_eff {res['l_eff']:.1f}", flush=True)


if __name__ == "__main__":
    main()
