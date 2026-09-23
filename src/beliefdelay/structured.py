"""Structured, CONTROLLABLE hidden-state cells.

Everything in models.py treats h_t as an opaque vector in R^d and asks, after training, whether it happens to encode
the belief (metrics.py), evolve like the true process (koopman.py), or stay stable (amplification.py). That is
measurement of a black box. This file builds two cells where h_t is, by construction, a point in a space with known
meaning, so it can be initialised, frozen, ablated and intervened on DIRECTLY -- control, not inference.

  KoopmanCell       h_t is a bank of k explicit complex modes z_k = r_k * e^{i theta_k}, parameterised in POLAR form.
                    r_k, theta_k ARE the model's Koopman spectrum -- not something fitted to h_t afterward (koopman.py
                    fits a spectrum TO a black box; here the spectrum IS the parameter). set_spectrum() initialises or
                    freezes it to env.koopman_eig() / env.closed_loop_eig() exactly; ablate_rotation() zeroes every
                    theta_k, mechanically removing the ability to rotate (a causal test of THEORY.md Prop 7, not a
                    correlational one).

  NeuralBayesCell   h_t IS a probability vector (log-domain) over n_state latent cells, and the recursion has the
                    exact FUNCTIONAL FORM of the Bayes filter (THEORY.md Prop 8: rho_t = diag(likelihood) T^T
                    rho_{t-1}) with LEARNED T, likelihood-map instead of the true ones. Because h_t lives in the same
                    space as the true belief, it can be compared to it with the SAME KL used everywhere else in this
                    project, with no probe at all. inject() resets h_t mid-rollout to the true belief -- a causal
                    intervention that asks whether the learned recursion behaves correctly GOING FORWARD from a known
                    starting point, which end-to-end loss alone cannot distinguish from lucky compensation.

Both match models.py's core interface: forward(x: (B,T,d_in)) -> (B,T,d_out), so `models.SeqModel` can host them
under new arch names ("koopman", "neuralbayes") and reuse every training/evaluation tool already built.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanCell(nn.Module):
    """k independent linear modes in polar form, each optionally input-selective (a la Mamba, but applied to a
    genuine complex mode so BOTH decay rate and rotation FREQUENCY can be modulated by the input, not just decay).

    State: z_t in C^k, tracked as two real channels (re, im) per mode. Update (per mode k, input scalar delta_t in
    (0,1] from a selective gate):
        z_{t,k} = r_k^{delta_t} * e^{i * delta_t * theta_k} * z_{t-1,k} + b_k(x_t)
    delta_t = 1 for every mode when selective=False (a PURE, non-input-dependent linear system -- the honest classical
    baseline: literally the matched Kalman/Koopman pair from lgssm.py, re-derived as a trainable module).
    """

    def __init__(self, d_model: int, k_modes: int | None = None, selective: bool = True):
        super().__init__()
        self.d = d_model
        self.k = k_modes or max(4, d_model // 2)
        self.selective = selective
        # polar parameterisation: r_k = sigmoid(raw_r) in (0,1) [always a contraction: no ablation can make it unstable],
        # theta_k free in R. THIS pair, not any derived quantity, is what set_spectrum/ablate_rotation touch directly.
        self.raw_r = nn.Parameter(torch.empty(self.k).uniform_(-2.0, 2.0))
        self.theta = nn.Parameter(torch.empty(self.k).uniform_(-np.pi, np.pi))
        self.Bre = nn.Linear(d_model, self.k, bias=False)
        self.Bim = nn.Linear(d_model, self.k, bias=False)
        self.readout = nn.Linear(2 * self.k, d_model)
        if selective:
            self.delta_proj = nn.Linear(d_model, self.k)

    # ---- CONTROL ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def set_spectrum(self, eig: np.ndarray, freeze: bool = False):
        """Initialise (or, with freeze=True, permanently fix) the k modes to given complex eigenvalues.  Extra slots
        beyond len(eig) keep their current values; fewer modes than len(eig) uses the len(eig) largest by magnitude."""
        eig = np.asarray(eig)
        eig = eig[np.argsort(-np.abs(eig))][: self.k]
        r = np.clip(np.abs(eig), 1e-4, 0.9999)
        theta = np.angle(eig)
        raw_r = np.log(r / (1 - r))                          # inverse sigmoid
        n = len(eig)
        self.raw_r[:n] = torch.tensor(raw_r, dtype=self.raw_r.dtype)
        self.theta[:n] = torch.tensor(theta, dtype=self.theta.dtype)
        self.raw_r.requires_grad_(not freeze)
        self.theta.requires_grad_(not freeze)

    @torch.no_grad()
    def ablate_rotation(self):
        """Zero every theta_k: mechanically removes rotation capacity while keeping decay magnitudes. A CAUSAL test
        of Prop 7 (a real-diagonal recurrence cannot carry rotation) -- run once with and once without this call."""
        self.theta.zero_()
        self.theta.requires_grad_(False)

    def spectrum(self) -> np.ndarray:
        r = torch.sigmoid(self.raw_r).detach().numpy()
        th = self.theta.detach().numpy()
        return r * np.exp(1j * th)

    # ---- forward -----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        r = torch.sigmoid(self.raw_r)                        # (k,) in (0,1): unconditionally stable
        theta = self.theta
        bre, bim = self.Bre(x), self.Bim(x)                  # (B,T,k) each
        if self.selective:
            delta = torch.sigmoid(self.delta_proj(x))         # (B,T,k) in (0,1): input-dependent time-warp
        else:
            delta = torch.ones(B, T, self.k, device=x.device, dtype=x.dtype)
        log_r_step = delta * torch.log(r).clamp_min(-30)      # (B,T,k)
        theta_step = delta * theta
        r_step, cos_t, sin_t = torch.exp(log_r_step), torch.cos(theta_step), torch.sin(theta_step)
        a_re, a_im = r_step * cos_t, r_step * sin_t           # per-step multiplier a_t = r_step * e^{i theta_step}
        re, im = torch.zeros(B, self.k, device=x.device, dtype=x.dtype), torch.zeros(B, self.k, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            re, im = a_re[:, t] * re - a_im[:, t] * im + bre[:, t], a_re[:, t] * im + a_im[:, t] * re + bim[:, t]
            outs.append(torch.cat([re, im], -1))
        return self.readout(torch.stack(outs, 1))


class NeuralBayesCell(nn.Module):
    """h_t is a genuine log-probability vector over n_state latent cells. The recursion has the exact functional
    form of the Bayes filter (Prop 8) with a LEARNED transition kernel T (n_state x n_state, row-softmax) and a
    LEARNED evidence map (input -> log-likelihood over the n_state cells). Because h_t = log(belief), it is decoded
    with NO probe: torch.softmax(h_t, -1) is a probability vector directly comparable to env's exact belief with the
    same KL used everywhere else. n_state defaults to the env's true state count for a direct, unpermuted comparison
    to env.T (accessible for inspection/initialisation via .T_logits), but any n_state is valid.
    """

    def __init__(self, d_model: int, n_state: int, d_out: int | None = None):
        super().__init__()
        self.n_state = n_state
        self.T_logits = nn.Parameter(torch.randn(n_state, n_state) * 0.1)     # learned log-transition (row-softmax)
        self.evidence = nn.Sequential(nn.Linear(d_model, 2 * n_state), nn.GELU(), nn.Linear(2 * n_state, n_state))
        self.log_prior = nn.Parameter(torch.zeros(n_state))
        self.readout = nn.Linear(n_state, d_out or d_model)
        self._injected: dict[int, torch.Tensor] = {}          # CONTROL: {position -> true belief (B, n_state)}

    # ---- CONTROL ---------------------------------------------------------------------------------------------
    def inject(self, position: int, true_belief: torch.Tensor) -> None:
        """Register a mid-rollout RESET: at this position, h_t is overwritten with log(true_belief) before continuing.
        A causal probe of the recursion itself -- does it behave correctly forward from a KNOWN-correct state, as
        opposed to end-to-end loss, which cannot distinguish a correct recursion from a lucky compensation elsewhere."""
        self._injected[position] = torch.log(true_belief.clamp_min(1e-8))

    def clear_injections(self) -> None:
        self._injected.clear()

    @torch.no_grad()
    def set_transition(self, T: np.ndarray, freeze: bool = False):
        """Initialise (or freeze) the learned transition kernel directly to the true env.T (row-stochastic)."""
        logT = torch.log(torch.tensor(T, dtype=self.T_logits.dtype).clamp_min(1e-8))
        self.T_logits.copy_(logT)
        self.T_logits.requires_grad_(not freeze)

    # ---- forward -----------------------------------------------------------------------------------------------
    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """Shared recursion: at each t, run the normal Bayes step (transition then evidence), THEN, if this position
        is injected, OVERWRITE the resulting belief -- so outs[t] is exactly the injected value when requested, and
        the OVERWRITTEN value (not the pre-injection one) is what the next step's transition sees as h_{t-1}. This
        was previously inverted (the injected value was fed in as the prior and one extra Bayes step was applied on
        top of it before being recorded), which silently shifted every injection by one timestep; caught by
        tests/test_structured.py::test_neuralbayes_injection_overwrites_exactly_at_its_position_and_only_there."""
        B, T, _ = x.shape
        logT = torch.log_softmax(self.T_logits, -1)           # (n_state, n_state): a genuine log-transition kernel
        h = self.log_prior.expand(B, self.n_state)
        outs = []
        for t in range(T):
            prior = torch.logsumexp(h.unsqueeze(-1) + logT, dim=1)     # log( softmax(h) @ T ), stable
            h = torch.log_softmax(prior + self.evidence(x[:, t]), -1)
            if t in self._injected:
                inj = self._injected[t]
                h = inj if inj.shape[0] == B else inj.expand(B, self.n_state)
            outs.append(h)
        return torch.stack(outs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self._run(x))

    def belief(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: the actual decoded belief sequence (B,T,n_state), softmax of the raw recursion (bypasses readout)."""
        return torch.softmax(self._run(x), -1)
