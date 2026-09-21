import numpy as np
import torch

from beliefdelay.grid import GRID_ENVS
from beliefdelay.metrics import excess_kl, fit_probe, fit_ridge_r2, belief_probes
from beliefdelay.pomdp import make_env
from beliefdelay.tuning import adaptive_lr_search


def test_test_vector_first_block_matches_pred_for_taken_action():
    for name in ("delayed_readout", "grid_patrol"):
        env = make_env(name)
        obs, act, _ = env.sample(6, 12, torch.Generator().manual_seed(0))
        post, pred = env.filter(obs, act)
        tv = env.test_vector(post, obs, K=3)
        M = env.n_out
        for a in range(env.A):
            blk = tv[..., a * 3 * M: a * 3 * M + M]              # k=1 block of action a
            m = act == a
            assert torch.allclose(blk[m], pred[m], atol=1e-10)
        nb = env.A * 3
        assert torch.allclose(tv.reshape(6, 12, nb, M).sum(-1), torch.ones(6, 12, nb, dtype=tv.dtype), atol=1e-9)


def test_moments_shapes_and_ring_conc_bounds():
    env = GRID_ENVS["grid_patrol"]()
    obs, act, _ = env.sample(4, 10, torch.Generator().manual_seed(1))
    post, _ = env.filter(obs, act)
    m = env.moments(post, obs)
    assert m.shape[-1] == len(env.moment_names) == 6
    assert (m[..., -1] >= -1e-12).all() and (m[..., -1] <= 1 + 1e-12).all()


def test_skill_is_one_for_bayes_and_zero_for_marginal():
    env = make_env("grid_drift")
    obs, act, _ = env.sample(64, 40, torch.Generator().manual_seed(2))
    post, pred = env.filter(obs, act)
    marg = pred[:, 10:].reshape(-1, 3).mean(0)
    d = dict(pred=pred, p_model=pred.clone())
    assert abs(excess_kl(d, 10)["skill"] - 1.0) < 1e-9
    d["p_model"] = marg.expand_as(pred).clone()
    assert abs(excess_kl(d, 10)["skill"]) < 1e-9


def test_probe_detects_signal_and_reports_train_test_gap():
    torch.manual_seed(0)
    X = torch.randn(3000, 8)
    Y = torch.softmax(X[:, :4] * 2, 1)
    r = fit_probe(X[:2000], Y[:2000], X[2000:], Y[2000:])
    assert r["score"] > 0.8 and abs(r["score"] - r["train_score"]) < 0.1
    rn = fit_probe(torch.randn(3000, 8)[:2000], Y[:2000], torch.randn(1000, 8), Y[2000:])
    assert rn["score"] < 0.05                               # noise features -> ~0 (chance)
    R2 = fit_ridge_r2(X[:2000], X[:2000, :2] * 3, X[2000:], X[2000:, :2] * 3)
    assert (R2 > 0.99).all()


def test_lr_search_extends_at_edge_and_defaults_when_flat():
    f = lambda lr: (np.log10(lr) - np.log10(0.05)) ** 2       # optimum at 0.05, above the initial grid
    r = adaptive_lr_search(f, [1e-3, 3e-3, 1e-2], log=lambda *_: None)
    assert r["best"] > 0.01 and r["extended"] and not r["flat"]
    r2 = adaptive_lr_search(lambda lr: 1.0 + 1e-4 * np.log(lr), [1e-3, 3e-3, 1e-2], default=3e-3, log=lambda *_: None)
    assert r2["flat"] and r2["best"] == 3e-3 and not r2["extended"]
