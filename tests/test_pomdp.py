"""Tests that the mathematics is right (exact filter vs brute force, rank bounds, exact finite windows)."""
import itertools

import numpy as np
import pytest
import torch

from beliefdelay.metrics import influence_profiles, kl, profile_stats, evaluate_model
from beliefdelay.pomdp import POMDP, DT, delayed_readout, random_pomdp, make_env


def tiny(A=2, N=3, M=2, seed=0):
    return random_pomdp(N=N, A=A, M=M, mix=0.2, stay=0.3, alpha_T=1.0, alpha_O=1.0, seed=seed)


def brute_force(env: POMDP, obs, act):
    """Enumerate ALL hidden paths.  Returns post_t, pred_t, and log P(o_{0:L-1}) for one sequence."""
    L = len(act)
    N = env.N
    post = np.zeros((L, N)); pred = np.zeros((L, env.M))
    T, O, mu0 = env.T.numpy(), env.O.numpy(), env.mu0.numpy()

    def joint_over_paths(t):  # weights over (s_0..s_t) given o_{0:t}, a_{0:t-1}
        w = {}
        for path in itertools.product(range(N), repeat=t + 1):
            p = mu0[path[0]] * O[0, path[0], obs[0]]
            for i in range(1, t + 1):
                p *= T[act[i - 1], path[i - 1], path[i]] * O[act[i - 1], path[i], obs[i]]
            w[path] = p
        return w

    for t in range(L):
        w = joint_over_paths(t)
        Z = sum(w.values())
        for path, p in w.items():
            post[t, path[-1]] += p / Z
        ps = post[t] @ T[act[t]]
        pred[t] = ps @ O[act[t]]
    Zfull = sum(joint_over_paths(L - 1).values())
    return post, pred, np.log(Zfull)


@pytest.mark.parametrize("seed", [0, 1])
def test_filter_matches_brute_force(seed):
    env = tiny(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    obs, act, _ = env.sample(3, 4, gen)
    post, pred, ll = env.filter(obs, act, return_loglik=True)
    for b in range(3):
        bp, bd, bl = brute_force(env, obs[b].tolist(), act[b].tolist())
        assert np.allclose(post[b].numpy(), bp, atol=1e-10)
        assert np.allclose(pred[b].numpy(), bd, atol=1e-10)
        assert abs(float(ll[b]) - bl) < 1e-10


def test_beliefs_are_distributions():
    env = make_env("random_slow")
    obs, act, _ = env.sample(8, 30, torch.Generator().manual_seed(0))
    post, pred = env.filter(obs, act)
    assert torch.allclose(post.sum(-1), torch.ones(8, 30, dtype=DT))
    assert torch.allclose(pred.sum(-1), torch.ones(8, 30, dtype=DT))
    assert (post >= 0).all() and (pred >= 0).all()


def test_hankel_rank_le_num_states_and_matches_filter():
    env = tiny(A=1, N=3, M=3)
    H = env.hankel(3, 3)  # 27 x 27
    r = int(torch.linalg.matrix_rank(H, atol=1e-10))
    assert r <= env.N                                     # Hankel/OOM rank bound
    # chain rule: P(u v) = P(u) P(v|u)  -> filter log-lik equals log string probability
    obs = torch.tensor([[0, 2, 1, 1]]); act = torch.zeros(1, 3, dtype=torch.long)
    _, _, ll = env.filter(obs, act, return_loglik=True)
    assert abs(float(ll[0]) - np.log(env.string_prob([0, 2, 1]))) < 1e-12


def test_dobrushin_bound():
    for mix in (0.1, 0.5):
        assert random_pomdp(mix=mix, stay=0.0, seed=3).dobrushin() <= 1 - mix + 1e-9


def test_window_filter_converges_and_equals_full_when_covering():
    env = make_env("random_slow")
    obs, act, _ = env.sample(64, 48, torch.Generator().manual_seed(1))
    _, pred = env.filter(obs, act)
    errs = []
    for W in (1, 4, 8, 16, 24):
        _, pw = env.window_filter(obs, act, W)
        errs.append(float(kl(pred[:, 24:], pw[:, 24:]).mean()))
    assert all(a > b for a, b in zip(errs, errs[1:]))     # finite-memory error decays with W
    _, pw = env.window_filter(obs, act, 47)
    assert torch.allclose(pw[:, :47], pred[:, :47])       # t < W: window covers the whole history
    assert not torch.allclose(pw[:, 47], pred[:, 47])      # t = W: o_0 has just fallen out of the window


def test_delayed_readout_is_exactly_a_delay_embedding():
    """THEORY Prop. 3: window of max(lags)+1 pairs gives the EXACT Bayes predictor; shorter does not."""
    env = delayed_readout()
    lmax = max(env.meta["lags"])
    obs, act, _ = env.sample(64, 40, torch.Generator().manual_seed(2))
    _, pred = env.filter(obs, act)
    err = lambda W: float(kl(pred[:, 20:], env.window_filter(obs, act, W)[1][:, 20:]).mean())
    assert err(lmax + 1) < 1e-12
    assert err(lmax) > 1e-3


class Oracle(torch.nn.Module):
    """Model that outputs the Bayes predictive law and the belief as its 'hidden state'."""
    def __init__(self, env):
        super().__init__(); self.env = env; self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        A = self.env.A
        obs = torch.cat([tokens // A, torch.zeros(tokens.shape[0], 1, dtype=torch.long)], 1)
        act = tokens % A
        post, pred = self.env.filter(obs, act)
        return pred.clamp_min(1e-300).log().float(), post.float()


def test_oracle_is_perfect_under_every_metric():
    env = tiny(N=4, M=3)
    cfg = dict(L=24, burn_in=8, kmax=6, eval_seqs=256, influence_seqs=64)
    res = evaluate_model(Oracle(env), env, cfg, eps_curve={1: 0.1, 2: 0.01, 4: 1e-4})
    assert res["excess_kl"] < 1e-9
    assert res["prof_w1"] < 1e-4   # oracle emits float32
    assert res["probe_belief_score"] > 0.95  # softmax-linear probe cannot reproduce a belief exactly (L2 + softmax)
    assert res["l_eff"] >= 4 or res["l_eff"] == 4.0


def test_profile_stats_sanity():
    Is = np.array([1.0, 0.0, 0.5, 0.0, 0.0, 1.0])
    assert profile_stats(Is, Is)["prof_w1"] == 0.0
    decay = np.exp(-np.arange(6.0))
    assert profile_stats(decay, Is)["prof_w1"] > 0.5
