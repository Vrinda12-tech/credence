import itertools

import numpy as np
import torch

from beliefdelay.grid import GridPOMDP, GRID_ENVS
from beliefdelay.metrics import kl
from beliefdelay.pomdp import DT

TINY = ["...", ".#.", "..."]


def brute(env, obs, act):
    """Enumerate every object path over all cells.  Independent of the recursive filter."""
    L = len(act); C = env.C
    pos, rd = (obs // env.R).tolist(), (obs % env.R).tolist()
    T, mu0, Lk = env.T.numpy(), env.mu0.numpy(), env.Lk.numpy()
    post = np.zeros((L, C)); pred = np.zeros((L, env.R))
    for t in range(L):
        w = np.zeros(C)
        for path in itertools.product(range(C), repeat=t + 1):
            p = mu0[path[0]] * Lk[pos[0], rd[0], path[0]]
            for i in range(1, t + 1):
                p *= T[path[i - 1], path[i]] * Lk[pos[i], rd[i], path[i]]
            w[path[-1]] += p
        post[t] = w / w.sum()
        npos = int(env.nxt[act[t], pos[t]])
        pred[t] = np.einsum("c,rc->r", post[t] @ T, Lk[npos])
    return post, pred


def test_filter_matches_brute_force_all_kinds():
    for kind in ("static", "drift"):
        env = GridPOMDP(layout=TINY, kind=kind)
        obs, act, _ = env.sample(2, 3, torch.Generator().manual_seed(0))
        post, pred = env.filter(obs, act)
        for b in range(2):
            bp, bd = brute(env, obs[b], act[b])
            assert np.allclose(post[b].numpy(), bp, atol=1e-10) and np.allclose(pred[b].numpy(), bd, atol=1e-10)


def test_positions_are_consistent_with_actions_and_walls():
    env = GRID_ENVS["grid_drift"]()
    obs, act, sts = env.sample(32, 40, torch.Generator().manual_seed(1))
    pos = env.pos_of(obs)
    assert torch.equal(pos[:, 1:], env.nxt[act, pos[:, :-1]])
    assert env.free.reshape(-1)[pos.numpy()].all() and env.free.reshape(-1)[sts.numpy()].all()


def test_sensor_rows_are_distributions_and_belief_normalised():
    env = GRID_ENVS["grid_patrol"]()
    assert torch.allclose(env.Lk.sum(1), torch.ones(env.C, env.C, dtype=DT))
    obs, act, _ = env.sample(8, 30, torch.Generator().manual_seed(2))
    post, pred = env.filter(obs, act)
    assert torch.allclose(post.sum(-1), torch.ones(8, 30, dtype=DT)) and torch.allclose(pred.sum(-1), torch.ones(8, 30, dtype=DT))


def test_patrol_belief_is_delay_shifted_evidence():
    """THEORY Prop 6: with p_move=1, log b_t(route[j]) = sum_k log e_{t-k}((j-k) mod P) + const  (lag-k evidence shifted by k)."""
    env = GridPOMDP(kind="patrol", p_move=1.0)
    P = len(env.route)
    obs, act, _ = env.sample(4, 25, torch.Generator().manual_seed(3))
    post, _ = env.filter(obs, act)
    pos, rd = env.pos_of(obs), obs % env.R
    for b in range(4):
        for t in (5, 24):
            logb = torch.zeros(P, dtype=DT)
            for j in range(P):
                for k in range(t + 1):
                    logb[j] += torch.log(env.Lk[pos[b, t - k], rd[b, t - k], env.route[(j - k) % P]])
            q = torch.softmax(logb, 0)
            assert torch.allclose(post[b, t, env.route], q, atol=1e-10)


def test_window_filter_error_decays_grid():
    env = GRID_ENVS["grid_drift"]()
    obs, act, _ = env.sample(64, 48, torch.Generator().manual_seed(4))
    _, pred = env.filter(obs, act)
    e = [float(kl(pred[:, 24:], env.window_filter(obs, act, W)[1][:, 24:]).mean()) for W in (1, 4, 12, 24)]
    assert all(a > b for a, b in zip(e, e[1:]))


def test_seeker_policy_is_valid_and_finds_object_more_often():
    rnd, seek = GRID_ENVS["grid_static"](), GridPOMDP(kind="static", policy="seeker")
    def hit_rate(env):
        obs, act, sts = env.sample(200, 40, torch.Generator().manual_seed(5))
        return float((env.pos_of(obs) == sts).any(1).float().mean())
    assert hit_rate(seek) > hit_rate(rnd)
    obs, act, _ = seek.sample(8, 20, torch.Generator().manual_seed(6))
    post, _ = seek.filter(obs, act)                       # filter remains valid under a history-dependent policy
    assert torch.allclose(post.sum(-1), torch.ones(8, 20, dtype=DT))
