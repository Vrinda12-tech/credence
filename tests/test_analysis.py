import numpy as np
from beliefdelay.analysis import boot_ci, holm, perm_test_less


def test_perm_exact_min_p():
    p = perm_test_less([0.1, 0.11, 0.12, 0.09, 0.1], [1, 1.1, 0.9, 1.2, 1.0])
    assert abs(p - 1 / 252) < 1e-12          # C(10,5)=252, perfectly separated


def test_perm_null_not_significant():
    rng = np.random.default_rng(0)
    assert perm_test_less(rng.normal(size=5), rng.normal(size=5)) > 0.02


def test_holm_monotone_and_bounded():
    adj = holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06]) and (adj <= 1).all()


def test_bootstrap_ci_contains_mean():
    m, lo, hi = boot_ci([1, 2, 3, 4, 5])
    assert lo <= m <= hi
