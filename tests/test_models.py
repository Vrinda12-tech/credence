import pytest
import torch

from beliefdelay.models import ARCHS, EXTRA_ARCHS, build_matched


@pytest.mark.parametrize("arch", ARCHS + EXTRA_ARCHS)
def test_shapes_causality_and_param_match(arch):
    torch.manual_seed(0)
    m = build_matched(arch, fields=(4, 3), n_out=4, target_params=40_000, max_len=32).eval()
    assert abs(m.n_params() - 40_000) / 40_000 < 0.12
    x = torch.randint(0, 12, (3, 24))  # fields (4,3) -> 12 codes
    lg, h = m(x)
    assert lg.shape == (3, 24, 4) and h.shape[:2] == (3, 24)
    x2 = x.clone()
    x2[:, 15:] = (x2[:, 15:] + 5) % 12
    lg2, _ = m(x2)
    # a causal model's outputs before position 15 must not depend on tokens >= 15
    assert torch.allclose(lg[:, :15], lg2[:, :15], atol=1e-4)
    assert not torch.allclose(lg[:, 15:], lg2[:, 15:], atol=1e-4)


def test_mambapy_matches_hf_mamba_in_function_class():
    """Both Mamba backends are causal selective SSMs of the same size class (sanity, not identity)."""
    a = build_matched("mamba", (4, 3), 4, 40_000, max_len=32)
    b = build_matched("mamba_hf", (4, 3), 4, 40_000, max_len=32)
    assert abs(a.n_params() - b.n_params()) / a.n_params() < 0.1


def test_continuous_input_and_delay_mlp_sees_exactly_W_lags():
    from beliefdelay.models import DELAY_WINDOW
    m = build_matched("delay_mlp", None, 1, 20_000, max_len=32, in_dim=1).eval()
    x = torch.randn(2, 30, 1)
    base = m(x)[0]
    x2 = x.clone(); x2[:, 10] += 5.0
    out = m(x2)[0]
    assert torch.allclose(base[:, :10], out[:, :10], atol=1e-6)                       # causal
    assert not torch.allclose(base[:, 10:10 + DELAY_WINDOW], out[:, 10:10 + DELAY_WINDOW], atol=1e-4)
    assert torch.allclose(base[:, 10 + DELAY_WINDOW:], out[:, 10 + DELAY_WINDOW:], atol=1e-5)   # forgets after W steps
