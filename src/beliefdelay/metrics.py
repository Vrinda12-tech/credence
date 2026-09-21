"""Faithfulness metrics.  Every one is defined against the EXACT Bayes filter of the POMDP.

M1  excess predictive KL     E_t KL( p*_t || p_theta,t )   = CE(theta) - H*        (>= 0, 0 iff Bayes-optimal)
M2  probe scores (PRIMARY = predictive-state "psr" probe; others secondary): belief-probe score       1 - KL(b*|probe) / KL(b*|marginal)                     (1 = belief perfectly decodable)
                             for two targets: predictive law p*_t (identifiable) and state belief b*_t
    (grid worlds add: map_mass_at_mode = Bayes mass on the probe's most-likely cell relative to Bayes' own mode;
     map_entropy_corr = correlation of decoded vs Bayes map uncertainty)
M3  memory-profile distance  W1 between normalised interventional influence profiles I_theta(k), I*(k)
     I(k) = E TV( p(.|history), p(.|history with o_{t-k} replaced by a different symbol) )
M4  effective delay depth    L_eff = window length W whose optimal window-filter error eps(W) equals the
                             model's excess KL (log-linear interpolation of the curve eps(W))

`model` is anything with  model(tokens)->(logits, hidden)  so an oracle can be plugged in for testing.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from .pomdp import POMDP

EPS = 1e-300


def _dev(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def tokens_of(env, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    return env.tokens(obs, act)


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p||q) over the last dim, float64, 0*log0 = 0."""
    p, q = p.to(torch.float64), q.to(torch.float64)
    return (p * (p.clamp_min(EPS).log() - q.clamp_min(EPS).log())).sum(-1)


def tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p.to(torch.float64) - q.to(torch.float64)).abs().sum(-1)


@torch.no_grad()
def collect(model, env: POMDP, n_seq: int, L: int, seed: int, chunk: int = 128):
    """Sample sequences, run model + exact filter.  Returns dict of tensors (all length-L axes)."""
    gen = torch.Generator().manual_seed(seed)
    obs, act, _ = env.sample(n_seq, L, gen)
    post, pred = env.filter(obs, act)
    toks = env.tokens(obs, act)
    logits, hid = [], []
    was_training = model.training
    model.eval()
    for i in range(0, n_seq, chunk):
        lg, h = model(toks[i:i + chunk].to(_dev(model)))
        logits.append(lg.to("cpu", torch.float64))
        hid.append(h.to("cpu", torch.float32))
    model.train(was_training)
    logits, hid = torch.cat(logits), torch.cat(hid)
    return dict(obs=obs, act=act, target=env.targets(obs), post=post, pred=pred, logits=logits, hidden=hid,
                p_model=torch.softmax(logits, -1))


def excess_kl(d: dict, burn_in: int) -> dict:
    p_star, p_mod = d["pred"][:, burn_in:], d["p_model"][:, burn_in:]
    k = kl(p_star, p_mod)
    ent = -(p_star * p_star.clamp_min(EPS).log()).sum(-1)
    marg = p_star.reshape(-1, p_star.shape[-1]).mean(0)          # law of the next target with NO memory at all
    klm = float(kl(p_star, marg.expand_as(p_star)).mean())
    return dict(kl_marginal=klm, skill=float(1.0 - k.mean() / max(klm, 1e-12)),
                excess_kl=float(k.mean()), excess_kl_sem=float(k.std() / np.sqrt(k.numel())),
                bayes_entropy=float(ent.mean()), tv_pred=float(tv(p_star, p_mod).mean()),
                ce=float((ent + k).mean()))


def heldout_ce(d: dict, burn_in: int) -> float:
    """Cross-entropy on SAMPLED next observations.  Needs no Bayes oracle => legitimate for model selection."""
    lp = torch.log_softmax(d["logits"][:, burn_in:], -1)
    tgt = d["target"][:, burn_in:]
    return float(-lp.gather(-1, tgt.unsqueeze(-1)).mean())


# ---------------------------------------------------------------- M2: linear softmax probe
def _block_logsoftmax(z, nb):
    return torch.log_softmax(z.reshape(z.shape[0], nb, -1), -1).reshape(z.shape[0], -1)


def _block_kl(y, q, nb):
    y, q = y.to(torch.float64), q.to(torch.float64)
    return (y * (y.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(1) / nb


def fit_probe(Xtr, Ytr, Xte, Yte, nblocks: int = 1, l2: float = 1e-3, max_iter: int = 150) -> dict:
    """Linear softmax probe (one softmax per block) minimising the mean block KL(y||q).  score = 1 - KL/KL(marginal)."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ytr, Yte = Ytr.float(), Yte.float()
    K = Ytr.shape[1]
    W = torch.zeros(Xtr.shape[1], K, requires_grad=True)
    c = torch.zeros(K, requires_grad=True)
    ent = -(Ytr * Ytr.clamp_min(1e-30).log()).sum(1).mean() / nblocks
    opt = torch.optim.LBFGS([W, c], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -(Ytr * _block_logsoftmax(Xtr @ W + c, nblocks)).sum(1).mean() / nblocks - ent + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        q_tr = _block_logsoftmax(Xtr @ W + c, nblocks).exp()
        q_te = _block_logsoftmax(Xte @ W + c, nblocks).exp()
        base = Ytr.mean(0, keepdim=True)
        kl_te, kl_tr = _block_kl(Yte, q_te, nblocks).mean().item(), _block_kl(Ytr, q_tr, nblocks).mean().item()
        kl_b = _block_kl(Yte, base.expand_as(Yte), nblocks).mean().item()
        kl_b_tr = _block_kl(Ytr, base.expand_as(Ytr), nblocks).mean().item()
    return dict(kl_probe=kl_te, kl_marginal=kl_b, score=1.0 - kl_te / max(kl_b, 1e-12),
                train_score=1.0 - kl_tr / max(kl_b_tr, 1e-12), q_test=q_te, y_test=Yte,
                predict=lambda X: _block_logsoftmax(((X - mu) / sd) @ W.detach() + c.detach(), nblocks).exp())


def fit_ridge_r2(Xtr, Ytr, Xte, Yte, lam: float = 1.0):
    """Ridge regression of low-dimensional belief moments on the hidden state; returns per-target held-out R^2."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = ((Xtr - mu) / sd).double(), ((Xte - mu) / sd).double()
    Ytr, Yte = Ytr.double(), Yte.double()
    ones = lambda X: torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    Xa = ones(Xtr)
    reg = lam * torch.eye(Xa.shape[1], dtype=Xa.dtype)
    reg[-1, -1] = 0
    Wt = torch.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Ytr)
    pred = ones(Xte) @ Wt
    sse = ((Yte - pred) ** 2).sum(0)
    sst = ((Yte - Ytr.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-12)
    return 1.0 - sse / sst


def belief_probes(d: dict, burn_in: int, env=None, K: int = 3, train_frac: float = 0.7) -> dict:
    """Probes of the last-layer hidden state.  PRIMARY: predictive-state probe (probe_psr).  Secondary: raw belief map,
    next-reading law, and low-dimensional belief moments (R^2).  train_score is reported to expose probe over-fitting."""
    H = d["hidden"][:, burn_in:]
    n = H.shape[0]
    ntr = int(n * train_frac)
    out = {}
    targets = [("probe_pred", d["pred"], 1), ("probe_belief", d["post"], 1)]
    if env is not None:
        targets.insert(1, ("probe_psr", env.test_vector(d["post"], d["obs"], K), env.A * K))
    for name, tgt, nb in targets:
        Y = tgt[:, burn_in:]
        f = lambda a, b: (H[a:b].reshape(-1, H.shape[-1]), Y[a:b].reshape(-1, Y.shape[-1]))
        (Xtr, Ytr), (Xte, Yte) = f(0, ntr), f(ntr, n)
        r = fit_probe(Xtr, Ytr, Xte, Yte, nblocks=nb)
        out[name + "_score"], out[name + "_kl"], out[name + "_train_score"] = r["score"], r["kl_probe"], r["train_score"]
        if name == "probe_belief":                       # spatial read-outs of the decoded belief MAP
            q, y = r["q_test"], r["y_test"]
            top = y.gather(1, q.argmax(1, keepdim=True)).squeeze(1)
            out["map_mass_at_mode"] = float((top / y.max(1).values.clamp_min(1e-12)).mean())
            h = lambda p: -(p * p.clamp_min(1e-30).log()).sum(1)
            hq, hy = h(q), h(y)
            out["map_entropy_corr"] = float(torch.corrcoef(torch.stack([hq, hy]))[0, 1]) if hy.std() > 0 else float("nan")
    if env is not None and hasattr(env, "moments"):
        M = env.moments(d["post"], d["obs"])[:, burn_in:].float()
        f = lambda a, b: (H[a:b].reshape(-1, H.shape[-1]), M[a:b].reshape(-1, M.shape[-1]))
        (Xtr, Ytr), (Xte, Yte) = f(0, ntr), f(ntr, n)
        r2 = fit_ridge_r2(Xtr, Ytr, Xte, Yte)
        out["probe_moments_r2"] = float(r2.mean())
        out["moments_r2_each"] = dict(zip(env.moment_names, [float(v) for v in r2]))
    return out


# ---------------------------------------------------------------- M3: memory profile
@torch.no_grad()
def influence_profiles(model, env: POMDP, kmax: int, n_seq: int, L: int, seed: int):
    """I(k) for k=0..kmax (lag k = position t-k, t=L-1), for the model and for the exact Bayes filter."""
    assert kmax < L - 1
    gen = torch.Generator().manual_seed(seed)
    obs, act, _ = env.sample(n_seq, L, gen)
    t = L - 1
    was_training = getattr(model, "training", False)
    model.eval()
    dv = _dev(model)
    lg, _ = model(env.tokens(obs, act).to(dv))
    p0 = torch.softmax(lg[:, t].to("cpu", torch.float64), -1)
    ps0 = env.filter(obs, act)[1][:, t]
    Im, Is = [], []
    for k in range(kmax + 1):
        delta = torch.randint(1, env.n_out, (n_seq,), generator=gen)
        o2 = env.perturb(obs, t - k, delta)
        lg2, _ = model(env.tokens(o2, act).to(dv))
        p1 = torch.softmax(lg2[:, t].to("cpu", torch.float64), -1)
        ps1 = env.filter(o2, act)[1][:, t]
        Im.append(float(tv(p0, p1).mean()))
        Is.append(float(tv(ps0, ps1).mean()))
    model.train(was_training)
    return np.array(Im), np.array(Is)


def profile_stats(Im: np.ndarray, Is: np.ndarray) -> dict:
    lags = np.arange(len(Im))
    pm, ps = Im / max(Im.sum(), 1e-12), Is / max(Is.sum(), 1e-12)
    w1 = float(wasserstein_distance(lags, lags, pm, ps))
    corr = float(np.corrcoef(Im, Is)[0, 1]) if Im.std() > 0 and Is.std() > 0 else float("nan")
    taps = Is >= 0.5 * Is.max()                        # lags the Bayes filter actually uses
    tap_mass = float(pm[taps].sum())

    def exp_r2(I):
        m = I > 1e-6
        if m.sum() < 3:
            return float("nan")
        y, x = np.log(I[m]), lags[m]
        a, b = np.polyfit(x, y, 1)
        res = ((y - (a * x + b)) ** 2).sum()
        tot = ((y - y.mean()) ** 2).sum() + 1e-12
        return float(1 - res / tot)

    return dict(prof_w1=w1, prof_corr=corr, prof_tap_mass=tap_mass, prof_sens_ratio=float(Im.sum() / max(Is.sum(), 1e-12)),
                prof_exp_r2_model=exp_r2(Im), prof_exp_r2_bayes=exp_r2(Is))


# ---------------------------------------------------------------- M4: effective delay depth
def window_error_curve(env: POMDP, Ws, n_seq: int, L: int, burn_in: int, seed: int) -> dict:
    """eps(W) = E KL(p* || p^W): excess KL of the best predictor restricted to the last W observations."""
    gen = torch.Generator().manual_seed(seed)
    obs, act, _ = env.sample(n_seq, L, gen)
    _, pred = env.filter(obs, act)
    out = {}
    for W in Ws:
        _, pw = env.window_filter(obs, act, W)
        out[int(W)] = float(kl(pred[:, burn_in:], pw[:, burn_in:]).mean())
    return out


def effective_depth(eps_curve: dict, model_excess: float, floor: float = 1e-12) -> float:
    """Smallest (interpolated) W with eps(W) <= model_excess.  inf if even the largest W is worse."""
    Ws = sorted(eps_curve)
    e = np.array([max(eps_curve[w], floor) for w in Ws])
    x = float(max(model_excess, floor))
    if x >= e[0]:
        return float(Ws[0])
    for i in range(1, len(Ws)):
        if e[i] <= x:
            w0, w1 = Ws[i - 1], Ws[i]
            f = (np.log(e[i - 1]) - np.log(x)) / (np.log(e[i - 1]) - np.log(e[i]) + 1e-30)
            return float(w0 + f * (w1 - w0))
    return float("inf")


def evaluate_model(model, env: POMDP, cfg: dict, eps_curve: dict | None = None, seed: int = 12345) -> dict:
    L, burn = cfg["L"], cfg["burn_in"]
    d = collect(model, env, cfg["eval_seqs"], L, seed)
    res = excess_kl(d, burn)
    res.update(belief_probes(d, burn, env, cfg.get("psr_K", 3)))
    Im, Is = influence_profiles(model, env, cfg["kmax"], cfg["influence_seqs"], L, seed + 1)
    res.update(profile_stats(Im, Is))
    res["profile_model"], res["profile_bayes"] = Im.tolist(), Is.tolist()
    if eps_curve is not None:
        res["l_eff"] = effective_depth(eps_curve, res["excess_kl"])
    return res
