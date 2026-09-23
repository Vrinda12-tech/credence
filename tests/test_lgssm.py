import numpy as np
import torch

from beliefdelay.lgssm import make_lgssm


def joint_conditional_mean(env, y):
    """Independent derivation: y_{0:T} is jointly Gaussian; E[y_{t+1}|y_{0:t}] = S_{t+1,0:t} S_{0:t}^-1 y_{0:t}."""
    A, c, P0, rn = env.Aa, env.ca, env.P0, env.rn
    T = len(y)
    cov = np.zeros((T, T))
    for s in range(T):
        for t in range(T):
            cov[s, t] = c @ (np.linalg.matrix_power(A, t - s) @ P0 if t >= s else P0 @ np.linalg.matrix_power(A.T, s - t)) @ c
    cov += rn ** 2 * np.eye(T)
    return [cov[t + 1, :t + 1] @ np.linalg.solve(cov[:t + 1, :t + 1], y[:t + 1]) for t in range(T - 1)]


def test_kalman_equals_gaussian_conditioning():
    for name in ("osc_rotating", "osc_decay"):
        env = make_lgssm(name)
        y, _ = env.sample(2, 9, torch.Generator().manual_seed(0))
        _, mu, _ = env.kalman(y)
        for b in range(2):
            assert np.allclose(mu[b].numpy(), joint_conditional_mean(env, y[b].numpy()), atol=1e-8)


def test_signal_variance_is_matched_and_excess_mse_identity():
    for name in ("osc_rotating", "osc_decay"):
        env = make_lgssm(name)
        assert abs(float(env.ca @ env.P0 @ env.ca) - 1.0) < 1e-9
    env = make_lgssm("osc_rotating")
    y, _ = env.sample(4000, 40, torch.Generator().manual_seed(1))
    _, mu, S = env.kalman(y)
    f = mu + 0.3                                            # a predictor that is off by a constant 0.3
    mse = ((f - y[:, 1:]) ** 2)[:, 10:].mean()
    assert abs(float(mse) - float(S[10:].mean() + 0.09)) < 0.03          # MSE = Bayes MSE + excess (Gaussian Prop 2)


def test_kernel_oscillates_iff_spectrum_is_complex():
    g_rot, g_dec = make_lgssm("osc_rotating").kernel(16), make_lgssm("osc_decay").kernel(16)
    assert (np.diff(np.sign(g_rot)) != 0).sum() >= 2
    assert (np.diff(np.sign(g_dec)) != 0).sum() == 0


def test_dmdc_recovers_closed_loop_spectrum_from_kalman_state():
    from beliefdelay.koopman import fit_dmdc, spectral_match
    for name in ("osc_rotating", "osc_decay"):
        env = make_lgssm(name)
        y, _ = env.sample(64, 120, torch.Generator().manual_seed(0))
        m_post, _, _ = env.kalman(y)
        fit = fit_dmdc(m_post, y[:, :-1, None], rank=2, burn=80)
        sm = spectral_match(fit["eig"], env.closed_loop_eig())
        assert sm["spec_dist"] < 1e-3 and fit["r2"] > 0.9999


def test_rotating_filter_has_complex_spectrum_decay_filter_is_real():
    assert np.abs(make_lgssm("osc_rotating").closed_loop_eig().imag).max() > 0.3
    assert np.abs(make_lgssm("osc_decay").closed_loop_eig().imag).max() < 1e-6
