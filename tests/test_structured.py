"""Correctness checks for structured.py's CONTROL mechanisms -- not a measurement suite. These just confirm the
control primitives do what they claim (spectrum set/freeze actually takes effect and stays frozen under
optimisation; injection actually overwrites the state at the right position and only the right position) so the
demonstrations in THEORY.md sec 13 can be trusted and extended without re-deriving them from scratch each time.
"""
import numpy as np
import torch
import torch.nn.functional as F

from beliefdelay.structured import KoopmanCell, NeuralBayesCell


def test_koopman_set_spectrum_is_exact_and_freeze_survives_a_gradient_step():
    torch.manual_seed(0)
    cell = KoopmanCell(d_model=8, k_modes=2, selective=False)
    true_eig = np.array([0.6 + 0.3j, 0.6 - 0.3j])
    cell.set_spectrum(true_eig, freeze=True)
    got = cell.spectrum()
    order = np.argsort(-got.imag)
    assert np.allclose(got[order], true_eig, atol=1e-4)
    x = torch.randn(4, 10, 8, requires_grad=False)
    out = cell(x)
    out.sum().backward()
    assert cell.raw_r.grad is None and cell.theta.grad is None         # frozen: no gradient at all
    with torch.no_grad():
        for p in cell.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
    assert np.allclose(cell.spectrum()[order], true_eig, atol=1e-4)    # unchanged after a step


def test_koopman_ablate_rotation_zeroes_theta_and_freezes_it():
    cell = KoopmanCell(d_model=8, k_modes=3, selective=False)
    cell.theta.data.copy_(torch.tensor([1.0, -2.0, 0.5]))
    cell.ablate_rotation()
    assert torch.allclose(cell.theta, torch.zeros(3))
    assert not cell.theta.requires_grad


def test_koopman_causal():
    cell = KoopmanCell(d_model=8, k_modes=4, selective=True)
    x = torch.randn(2, 12, 8)
    x2 = x.clone(); x2[:, 6:] += 5.0
    o1, o2 = cell(x), cell(x2)
    assert torch.allclose(o1[:, :6], o2[:, :6], atol=1e-5)
    assert not torch.allclose(o1[:, 6:], o2[:, 6:], atol=1e-4)


def test_neuralbayes_belief_is_a_probability_vector_and_causal():
    torch.manual_seed(0)
    cell = NeuralBayesCell(d_model=6, n_state=9)
    x = torch.randn(3, 15, 6)
    b = cell.belief(x)
    assert torch.allclose(b.sum(-1), torch.ones(3, 15), atol=1e-5)
    assert (b >= 0).all()
    x2 = x.clone(); x2[:, 8:] += 3.0
    b2 = cell.belief(x2)
    assert torch.allclose(b[:, :8], b2[:, :8], atol=1e-4)
    assert not torch.allclose(b[:, 8:], b2[:, 8:], atol=1e-3)


def test_neuralbayes_injection_overwrites_exactly_at_its_position_and_only_there():
    torch.manual_seed(0)
    cell = NeuralBayesCell(d_model=6, n_state=5)
    x = torch.randn(3, 10, 6)
    target = torch.softmax(torch.randn(3, 5), -1)
    base = cell.belief(x)
    cell.inject(4, target)
    injected = cell.belief(x)
    cell.clear_injections()
    assert torch.allclose(injected[:, 4], target, atol=1e-5)          # exact overwrite at the injected position
    assert torch.allclose(base[:, :4], injected[:, :4], atol=1e-6)    # the past is untouched (injection is causal)
    assert not torch.allclose(base[:, 4], injected[:, 4], atol=1e-3)  # and it actually changed something


def test_neuralbayes_set_transition_matches_env_T_exactly():
    T = np.array([[0.7, 0.3], [0.4, 0.6]])
    cell = NeuralBayesCell(d_model=4, n_state=2)
    cell.set_transition(T, freeze=True)
    learned = torch.softmax(cell.T_logits, -1).detach().numpy()
    assert np.allclose(learned, T, atol=1e-4)
    assert not cell.T_logits.requires_grad
