"""Grid-world search for a hidden object, with an EXACT belief map.

    * A map (walls '#', free '.'); an agent that moves on it (5 actions: up/down/left/right/stay; walls block).
    * The agent's own position is observed (it is part of the observation).
    * A hidden object lives on the map.  Its cell s_t is the hidden state; it evolves independently of the agent:
        static  : never moves                                   -> evidence accumulates forever
        drift   : lazy random walk on free cells                -> evidence decays (exponential forgetting)
        patrol  : moves along a fixed cyclic route w.p. p_move  -> a reading k steps ago constrains the object's
                  CURRENT cell shifted k steps along the route  (delay coordinates with a lag-dependent shift)
    * Sensor: a noisy proximity ping r in {silent, near, here} depending on Manhattan distance agent-object.
    * Belief state b_t(c) = P(object at c | positions, readings, actions so far): a probability MAP, computed exactly.

Observation symbol o_t = pos_t * R + r_t.  Token x_t = o_t * A + a_t.  Target: the next READING r_{t+1}
(the next position is a deterministic function of (pos_t, a_t) and is not part of the prediction target).

Because the object does not react to the agent, the Bayes filter stays exact for ANY action policy that is a function of
the history (including the belief-following 'seeker' policy below).
"""
from __future__ import annotations

import numpy as np
import torch

from .pomdp import DT

MAPS = {
    # 7x7 room with a 3x3 pillar; the 16 cells around the pillar form the patrol ring; 40 free cells.
    "pillar7": ["......." ,
                "......." ,
                "..###.." ,
                "..###.." ,
                "..###.." ,
                "......." ,
                "......."],
}

# P(reading | distance class).  Rows: d=0, d=1, d=2, d>=3.  Columns: silent, near, here.  All entries > 0.
SENSOR = np.array([[0.05, 0.15, 0.80],
                   [0.30, 0.65, 0.05],
                   [0.60, 0.39, 0.01],
                   [0.97, 0.029, 0.001]])
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up, down, left, right, stay


class GridPOMDP:
    def __init__(self, layout="pillar7", kind="patrol", name=None, p_move=0.85, stay=0.5,
                 policy="random", eps=0.3, sensor=SENSOR):
        rows = MAPS[layout] if isinstance(layout, str) else layout
        self.H, self.W = len(rows), len(rows[0])
        self.free = np.array([[ch != "#" for ch in row] for row in rows])
        C = self.H * self.W
        self.C, self.R, self.A_ = C, sensor.shape[1], len(MOVES)
        self.kind, self.policy, self.eps = kind, policy, eps
        self.name = name or f"grid_{kind}"
        self.rows = rows
        idx = lambda r, c: r * self.W + c
        free_flat = self.free.reshape(-1)
        self.free_cells = np.nonzero(free_flat)[0]

        # agent movement: nxt[a, cell]
        nxt = np.tile(np.arange(C), (self.A_, 1))
        for a, (dr, dc) in enumerate(MOVES):
            for r in range(self.H):
                for c in range(self.W):
                    if not self.free[r, c]:
                        continue
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < self.H and 0 <= c2 < self.W and self.free[r2, c2]:
                        nxt[a, idx(r, c)] = idx(r2, c2)
        self.nxt = torch.tensor(nxt, dtype=torch.long)

        # Manhattan distances and sensor likelihood Lk[pos, reading, object_cell]
        rr, cc = np.divmod(np.arange(C), self.W)
        dist = np.abs(rr[:, None] - rr[None]) + np.abs(cc[:, None] - cc[None])
        self.dist = torch.tensor(dist, dtype=torch.long)
        Lk = sensor[np.minimum(dist, 3)]                  # (C_pos, C_obj, R)
        self.Lk = torch.tensor(Lk, dtype=DT).permute(0, 2, 1).contiguous()  # (C_pos, R, C_obj)

        # object dynamics
        T = np.zeros((C, C))
        mu0 = np.zeros(C)
        self.route = None
        if kind == "static":
            T[np.arange(C), np.arange(C)] = 1.0
            mu0[self.free_cells] = 1.0
        elif kind == "drift":
            for c in self.free_cells:
                r, cl = divmod(c, self.W)
                nb = [idx(r + dr, cl + dc) for dr, dc in MOVES[:4]
                      if 0 <= r + dr < self.H and 0 <= cl + dc < self.W and self.free[r + dr, cl + dc]]
                T[c, c] += stay
                for n in nb:
                    T[c, n] += (1 - stay) / len(nb)
            mu0[self.free_cells] = 1.0
        elif kind == "patrol":
            self.route = self._ring(rows)
            P = len(self.route)
            for i, c in enumerate(self.route):
                T[c, self.route[(i + 1) % P]] += p_move
                T[c, c] += 1 - p_move
            mu0[self.route] = 1.0
        else:
            raise ValueError(kind)
        for c in range(C):                                 # walls: harmless self-loops (zero prior mass)
            if T[c].sum() == 0:
                T[c, c] = 1.0
        self.T = torch.tensor(T, dtype=DT)
        self.mu0 = torch.tensor(mu0 / mu0.sum(), dtype=DT)
        self.meta = dict(layout=layout if isinstance(layout, str) else "custom", kind=kind, p_move=p_move, stay=stay,
                         policy=policy, eps=eps)

    # ----------------------------------------------------------------- geometry
    def _ring(self, rows):
        """Cells adjacent (8-neighbourhood) to walls form the ring; ordered clockwise around the largest wall block."""
        wall = [(r, c) for r in range(self.H) for c in range(self.W) if not self.free[r, c]]
        r0, r1 = min(r for r, _ in wall) - 1, max(r for r, _ in wall) + 1
        c0, c1 = min(c for _, c in wall) - 1, max(c for _, c in wall) + 1
        cells = [(r0, c) for c in range(c0, c1)] + [(r, c1) for r in range(r0, r1)] + \
                [(r1, c) for c in range(c1, c0, -1)] + [(r, c0) for r in range(r1, r0, -1)]
        assert all(self.free[r, c] for r, c in cells)
        return [r * self.W + c for r, c in cells]

    # ----------------------------------------------------------------- interface shared with POMDP
    @property
    def N(self):
        return self.C

    @property
    def A(self):
        return self.A_

    @property
    def M(self):          # number of distinct next-reading targets
        return self.R

    @property
    def n_out(self):
        return self.R

    @property
    def token_fields(self):
        return (self.C, self.R, self.A_)

    def tokens(self, obs, act):
        return obs[:, :-1] * self.A_ + act

    def targets(self, obs):
        return obs[:, 1:] % self.R

    def perturb(self, obs, idx, delta):
        """Change the READING at time idx by `delta` (mod R); the agent position is left untouched."""
        o = obs.clone()
        pos, r = o[:, idx] // self.R, o[:, idx] % self.R
        o[:, idx] = pos * self.R + (r + delta) % self.R
        return o

    def pos_of(self, obs):
        return obs // self.R

    def test_vector(self, post, obs, K: int = 3):
        """Predictive-state vector: reading law k steps ahead when action a is repeated (positions follow the walls)."""
        pos0 = obs[:, :-1] // self.R
        outs = []
        for a in range(self.A_):
            prior, p = post, pos0
            for _ in range(K):
                prior = prior @ self.T
                p = self.nxt[a][p]
                outs.append(torch.einsum("blc,blrc->blr", prior, self.Lk[p]))
        return torch.cat(outs, -1)

    @property
    def moment_names(self):
        return ["E_row", "E_col", "entropy"] + (["ring_cos", "ring_sin", "ring_conc"] if self.route is not None else [])

    def moments(self, post, obs):
        """Expected row/col of the object, belief entropy, and (patrol) the first circular moment of the belief on the ring."""
        rr, cc = np.divmod(np.arange(self.C), self.W)
        rr, cc = torch.tensor(rr, dtype=DT), torch.tensor(cc, dtype=DT)
        cols = [post @ rr, post @ cc, -(post * post.clamp_min(1e-300).log()).sum(-1)]
        if self.route is not None:
            P = len(self.route)
            ang = 2 * np.pi * torch.arange(P, dtype=DT) / P
            pr = post[..., self.route]
            re, im = pr @ torch.cos(ang), pr @ torch.sin(ang)
            cols += [re, im, torch.sqrt(re ** 2 + im ** 2)]
        return torch.stack(cols, -1)

    def dobrushin(self) -> float:
        """delta(T_obj) over the cells the object can occupy (1.0 = no one-step contraction, e.g. static)."""
        sup = (self.mu0 > 0).nonzero(as_tuple=True)[0]
        P = self.T[sup][:, sup]
        return float((0.5 * (P[:, None, :] - P[None, :, :]).abs().sum(-1)).max())

    def stationary(self, iters: int = 5000, tol: float = 1e-13):
        P = 0.5 * (self.T + torch.eye(self.C, dtype=DT))
        pi = self.mu0.clone()
        for _ in range(iters):
            new = pi @ P
            if (new - pi).abs().max() < tol:
                pi = new
                break
            pi = new
        return pi / pi.sum()

    # ----------------------------------------------------------------- sampling (with optional belief-following policy)
    def sample(self, B: int, L: int, gen: torch.Generator):
        fc = torch.tensor(self.free_cells)
        pos = fc[torch.randint(len(fc), (B,), generator=gen)]
        s = torch.multinomial(self.mu0.expand(B, self.C), 1, generator=gen).squeeze(1)
        r = torch.multinomial(self.Lk[pos, :, s], 1, generator=gen).squeeze(1)
        obs, act, sts = [pos * self.R + r], [], [s]
        prior = self.mu0.expand(B, self.C).clone()
        for _ in range(L):
            un = prior * self.Lk[pos, r]
            b = un / un.sum(1, keepdim=True)
            a = torch.randint(self.A_, (B,), generator=gen)
            if self.policy == "seeker":
                target = b.argmax(1)
                d = self.dist[self.nxt[:, pos], target[None]]          # (A, B)
                greedy = d.argmin(0)
                use_rand = torch.rand(B, generator=gen) < self.eps
                a = torch.where(use_rand, a, greedy)
            s = torch.multinomial(self.T[s], 1, generator=gen).squeeze(1)
            pos = self.nxt[a, pos]
            r = torch.multinomial(self.Lk[pos, :, s], 1, generator=gen).squeeze(1)
            prior = b @ self.T
            act.append(a)
            obs.append(pos * self.R + r)
            sts.append(s)
        return torch.stack(obs, 1), torch.stack(act, 1), torch.stack(sts, 1)

    # ----------------------------------------------------------------- exact belief map
    def filter(self, obs, act, return_loglik=False):
        """post (B,L,C): belief map over object cell; pred (B,L,R): law of the next reading."""
        B, Lp1 = obs.shape
        L = Lp1 - 1
        pos, rd = obs // self.R, obs % self.R
        post = torch.zeros(B, L, self.C, dtype=DT)
        pred = torch.zeros(B, L, self.R, dtype=DT)
        ll = torch.zeros(B, dtype=DT)
        prior = self.mu0.expand(B, self.C).clone()
        for t in range(L):
            un = prior * self.Lk[pos[:, t], rd[:, t]]
            z = un.sum(1, keepdim=True)
            ll += torch.log(z.squeeze(1))
            b = un / z
            post[:, t] = b
            prior = b @ self.T
            npos = self.nxt[act[:, t], pos[:, t]]
            pred[:, t] = torch.einsum("bc,brc->br", prior, self.Lk[npos])
        return (post, pred, ll) if return_loglik else (post, pred)

    def window_filter(self, obs, act, W: int):
        """Optimal predictor that only sees the last W (position, reading) pairs, restarted from the stationary law."""
        post_f, pred_f = self.filter(obs, act)
        B, L, _ = post_f.shape
        if W >= L:
            return post_f, pred_f
        pos, rd = obs // self.R, obs % self.R
        ts = torch.arange(W, L)
        nt = len(ts)
        idx = ts[:, None] - (W - 1) + torch.arange(W)[None]
        p_w, r_w = pos[:, idx].reshape(B * nt, W), rd[:, idx].reshape(B * nt, W)
        prior = self.stationary().expand(B * nt, self.C).clone()
        for w in range(W):
            un = prior * self.Lk[p_w[:, w], r_w[:, w]]
            b = un / un.sum(1, keepdim=True)
            prior = b @ self.T
        npos = self.nxt[act[:, ts].reshape(-1), pos[:, ts].reshape(-1)]
        pr = torch.einsum("bc,brc->br", prior, self.Lk[npos])
        post, pred = post_f.clone(), pred_f.clone()
        post[:, W:] = b.reshape(B, nt, self.C)
        pred[:, W:] = pr.reshape(B, nt, self.R)
        return post, pred

    def render(self, pos=None, obj=None):
        """ASCII picture, for debugging."""
        out = []
        for r in range(self.H):
            line = ""
            for c in range(self.W):
                i = r * self.W + c
                line += "A" if pos == i else ("O" if obj == i else ("#" if not self.free[r, c] else "."))
            out.append(line)
        return "\n".join(out)


GRID_ENVS = {
    "grid_static": lambda: GridPOMDP(kind="static", name="grid_static"),
    "grid_drift": lambda: GridPOMDP(kind="drift", name="grid_drift", stay=0.5),
    "grid_patrol": lambda: GridPOMDP(kind="patrol", name="grid_patrol", p_move=0.85),
    # robustness variant: the agent follows its own belief (eps-greedy toward the belief mode)
    "grid_patrol_seek": lambda: GridPOMDP(kind="patrol", name="grid_patrol_seek", p_move=0.85, policy="seeker"),
}
