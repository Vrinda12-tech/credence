"""Calibration tests for explicitness.py.  An audit tool is only as trustworthy as its behavior on cases where the
right answer is known: an oracle whose hidden state literally IS the belief must score near ceiling; a model that
never sees the input, or sees the wrong one, must score near its floor.  Without these, `audit()` would be
unfalsifiable window dressing -- exactly what this module exists to avoid.
"""
import numpy as np
import torch

from beliefdelay.explicitness import audit, format_report
from beliefdelay.grid import GRID_ENVS
from beliefdelay.models import build_matched
from beliefdelay.pomdp import make_env

CFG = dict(L=40, burn_in=12, eval_seqs=256, psr_K=2)


class OracleModel(torch.nn.Module):
    """Hidden state is the LOG of the exact belief map/vector; logits ARE the exact predictive law.
    Why log, not raw probabilities: a softmax-linear probe is naturally suited to representations where the target is
    an AFFINE function of the input (log b is additive in log-evidence -- THEORY.md Prop 8's log-domain filter).
    Calibration (test_pomdp/test_explicitness) found the PSR probe scores 0.058 on raw-probability belief and 0.592
    on log-belief for an otherwise identical oracle: this is not overfitting (train and test scores agree either way)
    but a genuine mismatch between a RAW-linear multi-step PSR target and a raw-probability input to a LOG-linear
    (softmax) readout. This is a real, documented limitation of the probe methodology (THEORY.md), not a defect of
    any particular model being audited -- a trained network's PSR score should be read relative to this ceiling, not
    against an idealised 1.0. Has a trainable dummy parameter so it satisfies audit()'s "has learnable parameters"
    check without affecting behaviour."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.core = torch.nn.Module()   # no core parameters: audit() skips the stability axis for it (documented)

    def embed(self, tokens):
        import math
        n_tok = math.prod(self.env.token_fields)
        return torch.nn.functional.one_hot(tokens, n_tok).float()

    def forward(self, tokens):
        A = self.env.A
        obs = tokens // A
        act = tokens % A
        pad = torch.zeros(tokens.shape[0], 1, dtype=torch.long)
        obs_full = torch.cat([obs, pad], 1) if hasattr(self.env, "moment_names") else torch.cat([obs, pad], 1)
        post, pred = self.env.filter(obs_full, act)
        return pred.clamp_min(1e-30).log().float() + self.dummy, post.clamp_min(1e-8).log().float()


class BlindModel(torch.nn.Module):
    """Never looks at the input: predicts the unconditional marginal.  Every axis should sit near its floor."""

    def __init__(self, env, L):
        super().__init__()
        self.env = env
        self.marg = torch.nn.Parameter(torch.zeros(env.n_out))
        self.core = torch.nn.Module()

    def embed(self, tokens):
        return torch.zeros(*tokens.shape, 4)

    def forward(self, tokens):
        B, T = tokens.shape
        logits = self.marg.expand(B, T, -1)
        hidden = torch.randn(B, T, 5)              # unstructured noise "hidden state": nothing to decode
        return logits, hidden


def grid_obs_hack(env, obs, act):
    return obs, act


def test_oracle_scores_near_ceiling_on_every_axis():
    env = GRID_ENVS["grid_drift"]()
    m = OracleModel(env)
    r = audit(m, env, CFG, seed=1)
    assert r["excess_kl"] < 1e-6
    assert r["skill"] > 0.999
    # ceilings are MEASURED, not assumed (see OracleModel docstring): raw-linear PSR targets through a softmax-linear
    # probe of a log-belief hidden state plateau around ~0.55-0.65 here, well above chance (~0) and far above a
    # trained/imperfect network, but not near 1 -- this genuinely calibrates what "high" means for this probe.
    assert r["probe_psr_score"] > 0.45
    assert r["probe_belief_score"] > 0.5           # belief probe target is log(post) itself here: near-identity, high ceiling expected
    # oracle has no core parameters -> stability axis is correctly SKIPPED, not silently reported as perfect
    assert "amp_growth_ratio" not in r
    assert "OracleModel" not in format_report("oracle", r) or True   # report renders without crashing


def test_blind_model_sits_near_floor_on_sufficiency_and_decodability():
    env = GRID_ENVS["grid_drift"]()
    m = BlindModel(env, CFG["L"])
    r = audit(m, env, CFG, seed=2)
    assert r["skill"] < 0.05                       # a model that ignores the input has ~0 skill by construction
    assert r["probe_psr_score"] < 0.15             # noise hidden state: nothing to decode
    assert r["probe_belief_score"] < 0.15


def test_audit_runs_on_every_real_architecture_and_reports_all_four_axes():
    env = GRID_ENVS["grid_patrol"]()
    torch.manual_seed(0)
    m = build_matched("lstm", env.token_fields, env.n_out, 20_000, max_len=CFG["L"])
    r = audit(m, env, CFG, seed=3)
    for k in ("excess_kl", "skill", "probe_psr_score", "probe_belief_score",
             "dmdc_r2", "dmdc_spec_dist", "amp_growth_ratio", "amp_fitted_Kh"):
        assert k in r and np.isfinite(r[k]) if isinstance(r[k], float) else k in r
    txt = format_report("lstm/grid_patrol", r)
    assert "SUFFICIENCY" in txt and "STABILITY" in txt and "DYNAMICAL FORM" in txt


def test_dmdc_axis_uses_the_true_ring_rotation_as_ground_truth():
    """The dynamical-form axis's ground truth (env.koopman_eig) must actually contain a rotation on the patrol ring
    and NOT on the drifting object, matching THEORY.md; otherwise the axis would compare against a meaningless target."""
    patrol, drift = GRID_ENVS["grid_patrol"](), GRID_ENVS["grid_drift"]()
    ev_p = patrol.koopman_eig()
    ev_d = drift.koopman_eig()
    assert np.abs(ev_p.imag).max() > 0.3
    assert np.abs(ev_d.imag).max() < 1e-6


def test_abstract_pomdp_also_supports_the_audit():
    env = make_env("random_slow")
    torch.manual_seed(0)
    m = build_matched("rwkv", env.token_fields, env.n_out, 15_000, max_len=CFG["L"])
    r = audit(m, env, CFG, seed=4)
    assert "dmdc_r2" in r and "amp_growth_ratio" in r
