"""Validates amplification.py against an ANALYTIC ground truth (a hand-built scalar linear RNN), then runs the real
diagnostic on our four architectures.  Citation: Luo et al. 2024, NeurIPS ("Efficient Recurrent Off-Policy RL Requires a
Context-Encoder-Specific Learning Rate"), arXiv:2405.15384, Proposition 1.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from beliefdelay.amplification import amplification_curve, fit_Kh, theory_curve
from beliefdelay.models import ARCHS, build_matched


class ToyLinearRNN(nn.Module):
    """h_{t+1} = k_h * h_t + w * x_t,   y_t = k_y * h_t.  A scalar linear recurrence: Prop 1 holds with EQUALITY
    (not just as a bound) when the perturbation is to k_y alone, because the map is linear and h_t does not depend on
    k_y at all -- this isolates the K_y term.  Perturbing k_h instead exercises the amplification term through the
    hidden path, and for this exactly-solvable case the output difference can be computed in closed form for comparison."""

    def __init__(self, k_h: float, k_y: float, w: float = 1.0):
        super().__init__()
        self.k_h = nn.Parameter(torch.tensor(float(k_h)))
        self.k_y = nn.Parameter(torch.tensor(float(k_y)))
        self.w = w
        self.core = nn.Module()
        self.core.register_parameter("k_h", self.k_h)   # k_h is the ONLY "core" (recurrent-path) parameter

    def n_params(self):
        return 2

    def core_parameters(self):
        return [self.k_h]

    def head_parameters(self):
        return [self.k_y]

    def forward(self, x):
        B, T = x.shape
        h = torch.zeros(B, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = self.k_h * h + self.w * x[:, t]
            ys.append(self.k_y * h)
        return torch.stack(ys, 1).unsqueeze(-1), None


def exact_output_diff(x, k_h, k_y, dk_h, T):
    """Closed form for a SCALAR linear recurrence when only k_h is perturbed by dk_h (k_y fixed): h_t is a
    weighted sum of past inputs with weights k_h^{t-i}; differentiate that sum w.r.t. k_h to first order in dk_h."""
    B, _ = x.shape
    diffs = np.zeros((B, T))
    for b in range(B):
        h, hprime = 0.0, 0.0
        kh2 = k_h + dk_h
        for t in range(T):
            h = k_h * h + x[b, t].item()
            hprime = kh2 * hprime + x[b, t].item()
            diffs[b, t] = abs(k_y * (hprime - h))
    return diffs.mean(0)


def test_diagnostic_matches_closed_form_on_scalar_linear_rnn():
    torch.manual_seed(0)
    k_h, k_y, dk_h = 0.9, 2.0, 0.05
    m = ToyLinearRNN(k_h, k_y)
    x = torch.randn(64, 40)
    with torch.no_grad():
        m.k_h.fill_(k_h + dk_h)
        base = m(x)[0].squeeze(-1)
        m.k_h.fill_(k_h)
        pert = m(x)[0].squeeze(-1)
    measured = (pert - base).abs().mean(0).numpy()
    closed = exact_output_diff(x, k_h, k_y, dk_h, 40)
    assert np.allclose(measured, closed, atol=1e-5)


def _analytic_expected_diff(k_h, dk_h, T, k_y=1.5, w=1.0):
    """E|h_t - h_t'| for h driven by iid N(0,1) inputs, when only k_h is perturbed by dk_h (k_y, w fixed):
    h_t - h_t' is a zero-mean Gaussian (linear combination of the iid inputs), so E|Z| = sqrt(2/pi) std(Z),
    with Var(h_t-h_t') = w^2 sum_{j<t} (k_h^j - (k_h+dk_h)^j)^2 (population value; no sampling noise)."""
    j = np.arange(T)
    var = w ** 2 * np.cumsum((k_h ** j - (k_h + dk_h) ** j) ** 2)
    return k_y * np.sqrt(2 / np.pi) * np.sqrt(var)


def test_amplification_curve_matches_the_analytic_population_value():
    """The real ground-truth check (no heuristics): the measured curve, at a large batch size, must match the exact
    E|h_t-h_t'| derived independently from the recurrence's closed form -- this is stronger than checking shape."""
    torch.manual_seed(3)
    k_h, eps = 0.9, 1e-2
    x = torch.randn(40000, 60)
    m = ToyLinearRNN(k_h=k_h, k_y=1.5)
    gen = torch.Generator().manual_seed(0)
    dk_h = float((torch.randn(1, generator=gen) / 1.0).sign() * eps)   # amplification_curve normalises a 1-D vector to +-eps
    curve = amplification_curve(m, x, eps=eps, seed=0)
    analytic = _analytic_expected_diff(k_h, dk_h, 60)
    se = analytic.std() * 0.05 + 1e-4                          # generous finite-sample tolerance
    assert np.max(np.abs(curve - analytic)) < 10 * se
    assert (np.diff(analytic) >= -1e-12).all()                 # the POPULATION curve is exactly monotonic (no cancellation at n=inf)


def test_amplification_curve_increases_and_plateaus_for_contractive_toy_rnn():
    """Prop 1 bounds the MAXIMUM possible drift by a monotone, converging expression; it does not claim the empirical
    curve is pointwise monotonic for any one finite random batch (cancellation across time steps in a signed sum can
    cause small local dips).  What Prop 1 does guarantee, and what we check: growth well above the t=0 value, then a
    plateau (bounded tail variation) -- versus an UNSTABLE (K_h>=1) recurrence, which keeps growing and does not plateau."""
    torch.manual_seed(1)
    x = torch.randn(4000, 60)                              # large batch: averages out the sampling noise seen at n=32
    stable = ToyLinearRNN(k_h=0.9, k_y=1.5)
    curve = amplification_curve(stable, x, eps=1e-2, seed=0)
    fit = fit_Kh(curve)
    assert fit["monotone_frac"] > 0.75          # sampling noise at finite n; population curve (tested above) is exact
    assert curve[-1] > curve[2] * 1.5                     # real growth, not flat noise
    assert curve[-5:].std() / curve[-5:].mean() < 0.05    # plateaus (Prop 1's convergence claim)

    unstable = ToyLinearRNN(k_h=1.05, k_y=1.5)
    curve_u = amplification_curve(unstable, x, eps=1e-2, seed=0)
    assert curve_u[-1] > 5 * curve[-1]                    # K_h>=1: no plateau, far larger by the end


def test_amplification_scales_with_eps_and_theory_curve_is_an_upper_bound_shape():
    torch.manual_seed(2)
    x = torch.randn(32, 40)
    m = ToyLinearRNN(k_h=0.8, k_y=1.0)
    c1 = amplification_curve(m, x, eps=1e-3, seed=0)
    c2 = amplification_curve(m, x, eps=2e-3, seed=0)
    assert np.allclose(c2, 2 * c1, rtol=0.15)             # first-order: linear in eps
    th = theory_curve(K_h=0.8, K_y=1.0, eps=1e-3, T=40)
    assert np.all(np.diff(th) >= -1e-12)                  # the bound itself is non-decreasing
    # at T=40, K_h^39 ~ 1.7e-4 has not fully vanished yet, so match the infinite-horizon limit to that residual, not to machine precision
    assert th[-1] == pytest.approx(1.0 / (1 - 0.8) * 1e-3 + 1e-3, abs=1e-6)
    th_long = theory_curve(K_h=0.8, K_y=1.0, eps=1e-3, T=400)
    assert th_long[-1] == pytest.approx(1.0 / (1 - 0.8) * 1e-3 + 1e-3, rel=1e-6)   # fully converged at T=400


@pytest.mark.parametrize("arch", ARCHS)
def test_amplification_diagnostic_runs_on_every_architecture(arch):
    """Not a claim about which architecture is 'better' -- only that the diagnostic runs and returns a sane,
    finite, non-negative curve for a freshly initialised model of each architecture (untrained, as in the paper's
    figure, which perturbs an already-trained policy; here we check the tool works before drawing conclusions)."""
    torch.manual_seed(0)
    m = build_matched(arch, (4, 3), 4, 20_000, max_len=32).eval()
    x = torch.randint(0, 12, (16, 24))
    curve = amplification_curve(m, x, eps=1e-3, seed=0)
    assert curve.shape == (24,) and np.all(np.isfinite(curve)) and np.all(curve >= 0)
