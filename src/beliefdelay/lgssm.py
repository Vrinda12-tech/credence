"""Linear-Gaussian hidden-state systems where the Bayes-optimal belief is the KALMAN filter (classical, exact, closed form).

    x_{t+1} = A x_t + w_t,  w ~ N(0, Q)           hidden 2-D state (never observed)
    y_t     = c^T x_t + v_t, v ~ N(0, rn^2)       one noisy scalar sensor

Belief = N(m_t, P_t).  P_t follows a data-independent Riccati recursion; m_t is a LINEAR filter of the observations:
    mu*_t := E[y_{t+1} | y_{0:t}] = sum_k g(k) y_{t-k}     (g = the Kalman memory kernel, in steady state)
so the optimal memory kernel is g(k) = c^T A (A(I-KC))^k-type sums of exponentials: DAMPED OSCILLATION if the closed-loop
spectrum is complex (a rotating system: orbit, vortex shedding), monotone-exponential if it is real.

Matched pair (identical noise levels and stationary signal variance; only the spectrum of A differs):
    osc_rotating : A = r * Rot(omega), c = (1, 0)                    complex eigenvalues r e^{+-i omega}
    osc_decay    : A = diag(r1, r2),   c = (1, 1)/sqrt(2)            real eigenvalues

Exact identity (Gaussian analogue of Prop 2):  E (y_{t+1} - f)^2 = S_{t+1} + E (mu*_t - f)^2   for any predictor f of the history,
so   excess MSE := MSE - E S  =  E (mu*_t - f_t)^2.   Zero iff the model equals the Kalman predictor.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov

DT = torch.float64


class LinearGaussian:
    def __init__(self, A, c, rn: float = 0.5, signal_var: float = 1.0, name: str = "lgssm"):
        A = np.asarray(A, float)
        c = np.asarray(c, float)
        Q0 = np.eye(2)
        P0 = solve_discrete_lyapunov(A, Q0)                  # stationary covariance for unit process noise
        q2 = signal_var / float(c @ P0 @ c)                  # scale the process noise so Var(c^T x) = signal_var
        self.Aa, self.ca, self.Qa, self.rn = A, c, q2 * Q0, rn
        self.P0 = q2 * P0
        self.A, self.c, self.Q = (torch.tensor(v, dtype=DT) for v in (A, c, self.Qa))
        self.P0t = torch.tensor(self.P0, dtype=DT)
        self.name = name
        self.eig = np.linalg.eigvals(A)
        self.n_out, self.in_dim = 1, 1
        self.meta = dict(A=A.tolist(), c=c.tolist(), rn=rn, signal_var=signal_var, eig=[complex(e).__repr__() for e in self.eig])

    def sample(self, B: int, L: int, gen: torch.Generator):
        """y (B, L+1) float64, x (B, L+1, 2)."""
        Lc, Lq = torch.linalg.cholesky(self.P0t), torch.linalg.cholesky(self.Q)
        x = torch.randn(B, 2, generator=gen, dtype=DT) @ Lc.T
        xs, ys = [x], [x @ self.c + self.rn * torch.randn(B, generator=gen, dtype=DT)]
        for _ in range(L):
            x = x @ self.A.T + torch.randn(B, 2, generator=gen, dtype=DT) @ Lq.T
            xs.append(x)
            ys.append(x @ self.c + self.rn * torch.randn(B, generator=gen, dtype=DT))
        return torch.stack(ys, 1), torch.stack(xs, 1)

    def kalman(self, y: torch.Tensor):
        """Exact filter.  Returns m_post (B,L,2) = E[x_t|y_{0:t}],  mu (B,L) = E[y_{t+1}|y_{0:t}],  S (L,) = Var[y_{t+1}|y_{0:t}]."""
        B, Lp1 = y.shape
        L = Lp1 - 1
        A, c, Q = self.A, self.c, self.Q
        m_prior, P_prior = torch.zeros(B, 2, dtype=DT), self.P0t.clone()
        m_post, mu, S = torch.zeros(B, L, 2, dtype=DT), torch.zeros(B, L, dtype=DT), torch.zeros(L, dtype=DT)
        for t in range(L):
            s_t = c @ P_prior @ c + self.rn ** 2
            Kg = (P_prior @ c) / s_t
            m = m_prior + (y[:, t] - m_prior @ c)[:, None] * Kg[None]
            P_post = P_prior - torch.outer(Kg, c @ P_prior)
            m_post[:, t] = m
            m_prior = m @ A.T
            P_prior = A @ P_post @ A.T + Q
            mu[:, t] = m_prior @ c
            S[t] = c @ P_prior @ c + self.rn ** 2
        return m_post, mu, S

    def steady_gain(self, iters: int = 2000) -> torch.Tensor:
        P = self.P0t.clone()
        for _ in range(iters):
            s_t = self.c @ P @ self.c + self.rn ** 2
            Kg = (P @ self.c) / s_t
            P = self.A @ (P - torch.outer(Kg, self.c @ P)) @ self.A.T + self.Q
        return Kg

    def closed_loop_eig(self) -> np.ndarray:
        """Koopman spectrum of the FILTER (belief-mean dynamics driven by y): eig( A (I - K c^T) ).  Data-independent."""
        K = self.steady_gain()
        return np.linalg.eigvals((self.A @ (torch.eye(2, dtype=DT) - torch.outer(K, self.c))).numpy())

    def kernel(self, kmax: int, L: int = 64) -> np.ndarray:
        """Exact Kalman memory kernel g(k) = d mu*_t / d y_{t-k}, k=0..kmax (filter is linear, so an impulse gives it)."""
        out = []
        for k in range(kmax + 1):
            y = torch.zeros(1, L + 1, dtype=DT)
            y[0, L - 1 - k] = 1.0
            out.append(float(self.kalman(y)[1][0, L - 1]))
        return np.array(out)


def make_lgssm(name: str) -> LinearGaussian:
    if name == "osc_rotating":
        r, w = 0.97, 2 * np.pi / 8
        return LinearGaussian([[r * np.cos(w), -r * np.sin(w)], [r * np.sin(w), r * np.cos(w)]], [1.0, 0.0], name=name)
    if name == "osc_decay":
        return LinearGaussian([[0.97, 0.0], [0.0, 0.80]], [1 / np.sqrt(2), 1 / np.sqrt(2)], name=name)
    # long-memory pair: exposes finite-horizon approximation (real exponentials can mimic a short damped cosine; not a long one)
    if name == "osc_rotating_slow":
        r, w = 0.995, 2 * np.pi / 8
        return LinearGaussian([[r * np.cos(w), -r * np.sin(w)], [r * np.sin(w), r * np.cos(w)]], [1.0, 0.0], name=name)
    if name == "osc_decay_slow":
        return LinearGaussian([[0.995, 0.0], [0.0, 0.9]], [1 / np.sqrt(2), 1 / np.sqrt(2)], name=name)
    raise KeyError(name)


LG_ENVS = ["osc_rotating", "osc_decay"]
