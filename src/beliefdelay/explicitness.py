"""Explicitness audit: turns "the RNN learned it" from a trust claim into four separately measured, separately
falsifiable numbers, each checked against an exact ground truth computed elsewhere in this project. No axis implies
any other -- a model can be sufficient (predicts optimally) while being architecturally a black box (the information
is not linearly decodable, or its state does not evolve by anything resembling the process's own dynamics, or it is
unstable under small parameter perturbations).  That gap IS the finding a black-box "it converged" cannot see.

  SUFFICIENCY    excess KL / skill (metrics.py)            does h_t carry enough information to predict optimally?
  DECODABILITY   PSR / belief probes (metrics.py)          is that information LINEARLY exposed in h_t, or hidden
                                                             behind a nonlinearity only the model's own head can undo?
  DYNAMICAL FORM DMDc fit R^2 + spectral match (koopman.py) does h_t evolve approximately LINEARLY, and if so, does
                                                             its spectrum resemble the process's own Koopman spectrum
                                                             (env.koopman_eig / closed_loop_eig)?
  STABILITY      amplification growth_ratio (amplification.py)  does a small parameter change stay bounded across
                                                             rollout length, or does it keep growing ("memory drift")?

Each axis is independently falsifiable: a fresh RANDOM model, or a model given the WRONG target, must score near its
floor on that axis (see tests/test_explicitness.py); a model whose hidden state literally IS the belief (an oracle)
must score near ceiling on every axis except stability, which the audit does not run on the oracle since it has no
learnable parameters to perturb.
"""
from __future__ import annotations

import numpy as np
import torch

from .amplification import amplification_curve, fit_Kh
from .koopman import fit_dmdc, spectral_match
from .metrics import collect, excess_kl, belief_probes


def audit(model, env, cfg, seed: int = 2024, dmdc_rank: int = 6, amp_eps: float = 1e-2) -> dict:
    """Run all four axes on a (trained or fresh) model.  `env` needs `.filter`/`.sample` (POMDP/GridPOMDP interface)
    and, for the dynamical-form axis, a `.koopman_eig()` method (both POMDP and GridPOMDP have one)."""
    out: dict = {}

    # --- SUFFICIENCY + DECODABILITY: reuse the exact machinery from metrics.py -------------------------------
    d = collect(model, env, cfg.get("eval_seqs", 512), cfg["L"], seed)
    out.update(excess_kl(d, cfg["burn_in"]))
    out.update(belief_probes(d, cfg["burn_in"], env, cfg.get("psr_K", 3)))

    # --- DYNAMICAL FORM: does h_t evolve linearly, and does its spectrum match the process's own? --------------
    if hasattr(env, "koopman_eig"):
        H = d["hidden"]
        toks = env.tokens(d["obs"], d["act"])
        U = model.embed(toks).detach()                     # the real-valued signal actually driving the recurrence
        fit = fit_dmdc(H, U, rank=dmdc_rank, burn=cfg["burn_in"])
        eig_true = env.koopman_eig()
        eig_true = eig_true[np.argsort(-np.abs(eig_true))][1:dmdc_rank + 1]   # drop the trivial eigenvalue-1 mode
        sm = spectral_match(fit["eig"], eig_true)
        out["dmdc_r2"] = fit["r2"]
        out["dmdc_explained_var"] = fit["explained_var"]
        out["dmdc_spec_dist"] = sm["spec_dist"]
        out["dmdc_has_rotation"] = sm["has_rotation"]
        out["dmdc_true_has_rotation"] = bool(np.any(np.abs(eig_true.imag) > 1e-6))

    # --- STABILITY: does a small core-parameter perturbation stay bounded across rollout length? ---------------
    if hasattr(model, "core_parameters") and any(p.requires_grad for p in model.parameters()):
        toks = env.tokens(*env.sample(min(256, cfg.get("eval_seqs", 512)), cfg["L"], torch.Generator().manual_seed(seed + 1))[:2])
        curve = amplification_curve(model, toks, eps=amp_eps, seed=seed + 2)
        amp = fit_Kh(curve)
        out["amp_growth_ratio"] = amp["growth_ratio"]
        out["amp_fitted_Kh"] = amp["fitted_Kh"]
        out["amp_looks_contractive"] = amp["looks_contractive"]

    return out


def format_report(name: str, r: dict) -> str:
    lines = [f"=== {name} ==="]
    lines.append(f"  SUFFICIENCY    excess_kl={r.get('excess_kl', float('nan')):.4g}   skill={r.get('skill', float('nan')):.3f}")
    lines.append(f"  DECODABILITY   psr_probe={r.get('probe_psr_score', float('nan')):.3f}"
                 f"   belief_probe={r.get('probe_belief_score', float('nan')):.3f}")
    if "dmdc_r2" in r:
        lines.append(f"  DYNAMICAL FORM dmdc_r2={r['dmdc_r2']:.3f}   spec_dist={r['dmdc_spec_dist']:.3f}"
                     f"   rotation: model={r['dmdc_has_rotation']} true={r['dmdc_true_has_rotation']}")
    if "amp_growth_ratio" in r:
        lines.append(f"  STABILITY      growth_ratio={r['amp_growth_ratio']:.2f}   fitted_Kh={r['amp_fitted_Kh']:.3f}"
                     f"   contractive={r['amp_looks_contractive']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# CLI: audit every checkpoint in a run_experiment.py output directory.
#   python -m beliefdelay.explicitness results/cpu --env grid_patrol
# ---------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import glob
    import json
    import os

    from .grid import GRID_ENVS
    from .models import SeqModel
    from .pomdp import make_env

    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--env", default=None, help="restrict to one environment (default: all found)")
    ap.add_argument("--archs", nargs="+", default=None)
    a = ap.parse_args()

    pattern = os.path.join(a.results_dir, f"{a.env or '*'}__*__P*__s*.json")
    for f in sorted(glob.glob(pattern)):
        if "lr__" in f:
            continue
        meta = json.load(open(f))
        if a.archs and meta["arch"] not in a.archs:
            continue
        ck = os.path.join(a.results_dir, "ckpt", os.path.basename(f).replace(".json", ".pt"))
        if not os.path.exists(ck):
            print(f"(skip {os.path.basename(f)}: no checkpoint saved)")
            continue
        env = make_env(meta["env"])
        m = SeqModel(meta["arch"], env.token_fields, env.n_out, meta["d_model"], 2, max_len=meta["cfg"]["L"])
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        r = audit(m, env, meta["cfg"])
        print(format_report(f"{meta['env']}/{meta['arch']}/seed{meta['seed']}", r))
        print()
