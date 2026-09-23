"""Finite POMDPs with an EXACT Bayes filter, window (finite-memory) filters and OOM/Hankel tools.

Conventions (all tensors float64 so the filter is numerically exact):
  hidden state s_t in {0..N-1}, action a_t in {0..A-1}, observation o_t in {0..M-1}
  T[a, s, s']  = P(s_{t+1}=s' | s_t=s, a_t=a)
  O[a, s', o]  = P(o_{t+1}=o  | s_{t+1}=s', a_t=a)      (observation kernel conditioned on the
                                                          action that led into s')
  mu0[s]       = P(s_0 = s);   o_0 ~ O[0, s_0, .]         (action 0 is the "null action" for o_0)

Trajectory: o_0, (a_0, o_1), (a_1, o_2), ...  A model reads token x_t = (o_t, a_t) and must
predict o_{t+1}.  The Bayes-optimal quantities at time t are

  post_t(s)  = P(s_t = s            | o_{0:t}, a_{0:t-1})   (belief state)
  pred_t(o)  = P(o_{t+1} = o        | o_{0:t}, a_{0:t})      (Bayes-optimal predictive law)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch

DT = torch.float64


@dataclass
class POMDP:
    T: torch.Tensor
    O: torch.Tensor
    mu0: torch.Tensor
    name: str = "pomdp"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.T = self.T.to(DT)
        self.O = self.O.to(DT)
        self.mu0 = self.mu0.to(DT)
        assert self.T.ndim == 3 and self.T.shape[1] == self.T.shape[2]
        assert self.O.shape[:2] == self.T.shape[:2]
        assert torch.allclose(self.T.sum(-1), torch.ones(self.T.shape[:2], dtype=DT), atol=1e-9)
        assert torch.allclose(self.O.sum(-1), torch.ones(self.O.shape[:2], dtype=DT), atol=1e-9)
        assert abs(float(self.mu0.sum()) - 1.0) < 1e-9
        self.Ot = self.O.permute(0, 2, 1).contiguous()  # (A, M, N): Ot[a, o] = likelihood over s'

    # ---- sizes -------------------------------------------------------------------------
    @property
    def N(self):
        return self.T.shape[1]

    @property
    def A(self):
        return self.T.shape[0]

    @property
    def M(self):
        return self.O.shape[2]

    # ---- interface shared with GridPOMDP ---------------------------------------------------
    @property
    def n_out(self):
        return self.M

    @property
    def token_fields(self):
        return (self.M, self.A)

    def tokens(self, obs, act):
        return obs[:, :-1] * self.A + act

    def targets(self, obs):
        return obs[:, 1:]

    def perturb(self, obs, idx, delta):
        o = obs.clone()
        o[:, idx] = (o[:, idx] + delta) % self.M
        return o

    def test_vector(self, post, obs, K: int = 3):
        """Predictive-state ("PSR test") vector: for every action a and horizon k<=K the law of the observation k steps ahead
        when `a` is repeated.  A blockwise-simplex target (A*K blocks of size M).  This is what the belief MEANS for prediction,
        so it is identifiable where the raw state belief is not (THEORY.md, identifiability)."""
        outs = []
        for a in range(self.A):
            prior = post
            for _ in range(K):
                prior = prior @ self.T[a]
                outs.append(prior @ self.O[a])
        return torch.cat(outs, -1)

    def moments(self, post, obs):
        """Low-dimensional summaries of the belief: entropy only for abstract POMDPs."""
        h = -(post * post.clamp_min(1e-300).log()).sum(-1)
        return h[..., None]

    moment_names = ["entropy"]

    # ---- helpers -----------------------------------------------------------------------
    def _propagate(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Row-vector push-forward b @ T[a] with a per-row action (loops over A, not over rows)."""
        out = torch.empty_like(b)
        for ai in range(self.A):
            idx = (a == ai).nonzero(as_tuple=True)[0]
            if idx.numel():
                out[idx] = b[idx] @ self.T[ai]
        return out

    def stationary(self, iters: int = 5000, tol: float = 1e-13) -> torch.Tensor:
        """Stationary law of the action-averaged (lazy) chain; used as the window-filter prior."""
        P = 0.5 * (self.T.mean(0) + torch.eye(self.N, dtype=DT))
        pi = torch.full((self.N,), 1.0 / self.N, dtype=DT)
        for _ in range(iters):
            new = pi @ P
            if (new - pi).abs().max() < tol:
                pi = new
                break
            pi = new
        return pi / pi.sum()

    def koopman_eig(self) -> "np.ndarray":
        """Eigenvalues of the TRUE hidden-state transition operator (the process's own Koopman/Perron-Frobenius
        spectrum), action-averaged.  This is what a model's hidden-state dynamics is compared against in
        explicitness.py; it is the process's spectrum, not a filter closed-loop spectrum (no clean LTI filter form
        exists outside the linear-Gaussian case -- see lgssm.LinearGaussian.closed_loop_eig for that case)."""
        import numpy as np
        return np.linalg.eigvals(self.T.mean(0).numpy())

    def dobrushin(self) -> float:
        """max_a delta(T_a), delta(P)=max_{i,j} TV(P_i, P_j).  delta<1 => contraction of the predict step."""
        best = 0.0
        for a in range(self.A):
            P = self.T[a]
            d = 0.5 * (P[:, None, :] - P[None, :, :]).abs().sum(-1)
            best = max(best, float(d.max()))
        return best

    # ---- sampling ----------------------------------------------------------------------
    def sample(self, B: int, L: int, gen: torch.Generator):
        """Uniform-random action policy.  Returns obs (B, L+1), act (B, L), states (B, L+1)."""
        s = torch.multinomial(self.mu0.expand(B, self.N), 1, generator=gen).squeeze(1)
        a0 = torch.zeros(B, dtype=torch.long)
        o = torch.multinomial(self.O[a0, s], 1, generator=gen).squeeze(1)
        obs, act, sts = [o], [], [s]
        for _ in range(L):
            a = torch.randint(self.A, (B,), generator=gen)
            s = torch.multinomial(self.T[a, s], 1, generator=gen).squeeze(1)
            o = torch.multinomial(self.O[a, s], 1, generator=gen).squeeze(1)
            act.append(a)
            obs.append(o)
            sts.append(s)
        return torch.stack(obs, 1), torch.stack(act, 1), torch.stack(sts, 1)

    # ---- exact Bayes filter ------------------------------------------------------------
    def filter(self, obs: torch.Tensor, act: torch.Tensor, return_loglik: bool = False):
        """obs (B, L+1), act (B, L) -> post (B, L, N), pred (B, L, M) [, loglik of o_{0:L-1} (B,)]."""
        B, Lp1 = obs.shape
        L = Lp1 - 1
        post = torch.zeros(B, L, self.N, dtype=DT)
        pred = torch.zeros(B, L, self.M, dtype=DT)
        ll = torch.zeros(B, dtype=DT)
        prior = self.mu0.expand(B, self.N).clone()
        a_prev = torch.zeros(B, dtype=torch.long)
        for t in range(L):
            unnorm = prior * self.Ot[a_prev, obs[:, t]]
            z = unnorm.sum(1, keepdim=True)
            ll += torch.log(z.squeeze(1))
            b = unnorm / z
            post[:, t] = b
            a = act[:, t]
            prior = self._propagate(b, a)
            pred[:, t] = torch.einsum("bn,bnm->bm", prior, self.O[a])
            a_prev = a
        return (post, pred, ll) if return_loglik else (post, pred)

    # ---- finite-memory ("delay-embedding") filter --------------------------------------
    def window_filter(self, obs: torch.Tensor, act: torch.Tensor, W: int):
        """Bayes filter that only sees the last W observations (restarted from the stationary law).

        For t < W it coincides with the full filter (the window covers the whole history).
        This is the optimal predictor that uses a hard delay window of length W.
        """
        post_f, pred_f = self.filter(obs, act)
        B, L, _ = post_f.shape
        if W >= L:
            return post_f, pred_f
        ts = torch.arange(W, L)
        nt = len(ts)
        idx = ts[:, None] - (W - 1) + torch.arange(W)[None]  # (nt, W)
        o_w = obs[:, idx].reshape(B * nt, W)
        a_w = act[:, idx].reshape(B * nt, W)
        a_prev = act[:, ts - W].reshape(B * nt)
        prior = self.stationary().expand(B * nt, self.N).clone()
        for w in range(W):
            unnorm = prior * self.Ot[a_prev, o_w[:, w]]
            b = unnorm / unnorm.sum(1, keepdim=True)
            a = a_w[:, w]
            prior = self._propagate(b, a)
            a_prev = a
        pred = torch.einsum("bn,bnm->bm", prior, self.O[a_prev])
        post, pr = post_f.clone(), pred_f.clone()
        post[:, W:] = b.reshape(B, nt, self.N)
        pr[:, W:] = pred.reshape(B, nt, self.M)
        return post, pr

    # ---- observable-operator / Hankel tools (action-free use: uses action 0 only) ------
    def string_prob(self, seq) -> float:
        v = self.mu0 * self.O[0][:, seq[0]]
        for o in seq[1:]:
            v = (v @ self.T[0]) * self.O[0][:, o]
        return float(v.sum())

    def hankel(self, h: int, f: int) -> torch.Tensor:
        """H[u, v] = P(o-string u followed by o-string v), |u|=h, |v|=f.  rank(H) <= N."""
        pre = list(itertools.product(range(self.M), repeat=h))
        suf = list(itertools.product(range(self.M), repeat=f))
        H = torch.zeros(len(pre), len(suf), dtype=DT)
        for i, u in enumerate(pre):
            for j, v in enumerate(suf):
                H[i, j] = self.string_prob(list(u) + list(v))
        return H


# =====================================================================================
# Environment families
# =====================================================================================
def _noisy_channel(k: int, eps: float) -> np.ndarray:
    C = np.full((k, k), eps / (k - 1))
    np.fill_diagonal(C, 1.0 - eps)
    return C  # C[z, x] = P(x | z)


def delayed_readout(k: int = 2, D: int = 7, lags=(0, 2, 5), eps: float = 0.05) -> POMDP:
    """Shift-register POMDP whose Bayes filter is an exact delay embedding.

    Hidden state: register (z_0..z_{D-1}), z_0 newest, new symbol iid uniform on {0..k-1} every step.
    Action a picks a readout lag lags[a].  Observation o=(u,v) with
        u = noisy z_0 (the fresh cue),    v = noisy z_{lags[a]+1}  (an old symbol, read out).
    To predict v_{t+1} one must remember the cue u seen `lags[a_t]` steps ago, so the belief that
    matters is a function of the delay coordinates (u_t, u_{t-1}, ..., u_{t-D+1}).  See THEORY.md, Prop. 3:
    the Bayes-optimal predictor is EXACTLY a function of the last max(lags)+1 (obs, action) pairs.
    """
    lags = tuple(lags)
    assert max(lags) + 1 <= D - 1, "need lags[a]+1 <= D-1"
    N, A, M = k ** D, len(lags), k * k
    T = np.zeros((A, N, N))
    for s in range(N):
        base = k * (s % (k ** (D - 1)))
        for znew in range(k):
            T[:, s, base + znew] += 1.0 / k
    C = _noisy_channel(k, eps)

    def z(s, j):
        return (s // k ** j) % k

    O = np.zeros((A, N, M))
    for a, lag in enumerate(lags):
        for s in range(N):
            for u in range(k):
                for v in range(k):
                    O[a, s, u * k + v] = C[z(s, 0), u] * C[z(s, lag + 1), v]
    mu0 = np.full(N, 1.0 / N)
    return POMDP(torch.tensor(T), torch.tensor(O), torch.tensor(mu0), name="delayed_readout",
                 meta=dict(k=k, D=D, lags=lags, eps=eps))


def random_pomdp(N: int = 16, A: int = 3, M: int = 6, mix: float = 0.5, stay: float = 0.0,
                 alpha_T: float = 0.3, alpha_O: float = 0.3, seed: int = 0, name: str = "random") -> POMDP:
    """T_a = (1-mix) * (stay*I + (1-stay)*Dir_a) + mix * 1 nu^T.   Dobrushin coeff <= 1-mix.

    `mix` controls forgetting of the prior (fast/slow filter stability), `stay` adds sticky dynamics
    that lengthen the memory the Bayes filter needs.  O is shared across actions.
    """
    rng = np.random.default_rng(seed)
    T = np.zeros((A, N, N))
    nu = rng.dirichlet(np.ones(N))
    for a in range(A):
        Dm = rng.dirichlet(alpha_T * np.ones(N), size=N)
        T[a] = (1 - mix) * (stay * np.eye(N) + (1 - stay) * Dm) + mix * nu[None, :]
    Ob = rng.dirichlet(alpha_O * np.ones(M), size=N) + 1e-4
    Ob /= Ob.sum(1, keepdims=True)
    O = np.repeat(Ob[None], A, axis=0)
    mu0 = np.full(N, 1.0 / N)
    return POMDP(torch.tensor(T), torch.tensor(O), torch.tensor(mu0), name=name,
                 meta=dict(N=N, A=A, M=M, mix=mix, stay=stay, seed=seed))


ENVS = {
    "delayed_readout": lambda: delayed_readout(),
    "random_fast": lambda: random_pomdp(mix=0.5, stay=0.0, alpha_O=0.1, seed=1, name="random_fast"),
    "random_slow": lambda: random_pomdp(mix=0.01, stay=0.9, alpha_O=0.1, seed=2, name="random_slow"),
}


def make_env(name: str):
    from .grid import GRID_ENVS
    if name in GRID_ENVS:
        return GRID_ENVS[name]()
    return ENVS[name]()
