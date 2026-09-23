"""
Tests for  evaluation/rigorous_validation.py.

In research it is easy, and often unconscious, to bend an evaluation until p < 0.05: choose the tail after seeing the data, peek and
stop when it turns significant, report only the comparisons that worked, drop the runs that failed, move the margin.  This file proves,
by exact enumeration or by simulation with theory-derived tolerances, that the tools which judge our experiments cannot be tampered
with, misconfigured or misused.

Structure.  Each guard is stated as a THEOREM and programmed as a *property checker* that takes an implementation.  The checker must
PASS on the real implementation and FAIL on every deliberately broken "mutant" (a mutation test): if someone weakens a guard, or if a
checker is too lax to notice, the suite goes red.
"""
import itertools
import json
import copy
from math import comb

import numpy as np
import pytest
from scipy import stats

from beliefdelay.evaluation import rigorous_validation as rv

ALTS = ["two-sided", "less", "greater"]


# ===================================================================================================== helpers / mutants
def exact_T(x, y):
    pooled, nx = np.concatenate([x, y]), len(x)
    n = len(pooled)
    idx = np.array(list(itertools.combinations(range(n), nx)))
    sx = pooled[idx].sum(1)
    return sx / nx - (pooled.sum() - sx) / (n - nx), float(np.mean(x) - np.mean(y))


def mut_strict_inequality(x, y, alternative="two-sided", **kw):
    """MUTANT: counts only permutations STRICTLY more extreme than the observed one (the observed labelling excludes itself) -> p can be 0."""
    T, obs = exact_T(np.asarray(x, float), np.asarray(y, float))
    if alternative == "two-sided":
        return float((np.abs(T) > abs(obs) + 1e-12).mean())
    return float(((T > obs + 1e-12) if alternative == "greater" else (T < obs - 1e-12)).mean())


def mut_peek_direction(x, y, alternative="two-sided", **kw):
    """MUTANT: picks the favourable tail AFTER seeing the data but reports it as 'two-sided'."""
    return min(rv.permutation_pvalue(x, y, "less"), rv.permutation_pvalue(x, y, "greater"))


def mut_mc_no_plus_one(x, y, alternative="two-sided", n_perm=50, seed=0, **kw):
    """MUTANT: Monte-Carlo p-value b/m (can be exactly 0, which is never a valid p-value)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pooled, nx = np.concatenate([x, y]), len(x)
    rng = np.random.default_rng(seed)
    obs = abs(x.mean() - y.mean())
    b = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        b += abs(p[:nx].mean() - p[nx:].mean()) >= obs - 1e-12
    return b / n_perm


def mut_holm_none(p):
    return np.asarray(p, float)


def mut_holm_off_by_one(p):
    p = np.asarray(p, float)
    return np.minimum(1.0, max(len(p) - 1, 1) * p)


def mut_holm_no_monotone(p):
    """MUTANT: forgets the running maximum, so adjusted p-values are not monotone in the sorted order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    for r, i in enumerate(order):
        adj[i] = min(1.0, (len(p) - r) * p[i])
    return adj


# ===================================================================================================== property checkers (theorems)
def orbit_pvalues(pfunc, data, nx, alternative):
    n, out = len(data), []
    for c in itertools.combinations(range(n), nx):
        m = np.zeros(n, bool)
        m[list(c)] = True
        out.append(pfunc(data[m], data[~m], alternative))
    return np.array(out)


DATA8 = np.array([0.31, 1.7, -0.4, 2.2, 0.9, -1.3, 0.05, 1.1])


def type1_exact_ok(pfunc, alternative="two-sided", nx=4, data=DATA8):
    """THEOREM 1 (finite-sample validity).  Under exchangeability every one of the C(n,k) relabellings of the SAME data is equally likely,
    so a valid p satisfies  P(p <= a) <= a  for EVERY a.  Checked on the full permutation orbit: no Monte-Carlo error at all."""
    ps = orbit_pvalues(pfunc, data, nx, alternative)
    for a in list(np.unique(ps)) + [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9]:
        if np.mean(ps <= a + 1e-12) > a + 1e-12:
            return False
    return True


def mc_pvalue_positive_ok(pfunc):
    """THEOREM 2 (Phipson & Smyth).  A Monte-Carlo permutation p-value must be (b+1)/(m+1) >= 1/(m+1): p = 0 is never valid."""
    x, y = np.arange(6.0) + 10, np.arange(6.0)                  # perfectly separated
    return pfunc(x, y, "two-sided", n_perm=40, seed=1, max_exact=0) >= 1 / 41 - 1e-12


def holm_fwer_ok(holmfunc, m, K, alpha, dependence="independent"):
    """THEOREM 3 (Holm 1979).  Under the complete null, with p-values that are (super-)uniform on a discrete grid {1/K..1},
    P(any adjusted p <= alpha) <= alpha.  Exact enumeration of the joint distribution; also for adversarial dependence."""
    grid = np.arange(1, K + 1) / K
    if dependence == "independent":
        outcomes = itertools.product(grid, repeat=m)
    elif dependence == "comonotone":
        outcomes = ((g,) * m for g in grid)
    elif dependence == "disjoint":                                # worst case for FWER: rejection events as disjoint as possible (m=2)
        outcomes = ((grid[k], grid[(k + K // 2) % K]) for k in range(K))
    else:
        raise ValueError
    hits = tot = 0
    for o in outcomes:
        tot += 1
        hits += bool(np.min(holmfunc(list(o))) <= alpha + 1e-12)
    return hits / tot <= alpha + 1e-12


def holm_structure_ok(holmfunc, trials=200, seed=0):
    """Holm's algebra: adj >= p, adj in [0,1], adj_holm <= adj_bonf, min(adj) = min(1, m*p_min), order-preserving, permutation-equivariant."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        m = int(rng.integers(1, 9))
        p = np.round(rng.random(m), 3)
        a = np.asarray(holmfunc(p))
        if not (np.all(a >= p - 1e-12) and np.all(a <= 1 + 1e-12) and np.all(a <= np.minimum(1, m * p) + 1e-12)):
            return False
        if not np.isclose(a.min(), min(1.0, m * p.min())):                # the smallest p pays the full price m (step 1 of the step-down)
            return False
        o = np.argsort(p, kind="stable")
        if np.any(np.diff(a[o]) < -1e-12):                            # monotone: smaller p never gets a larger adjusted p
            return False
        perm = rng.permutation(m)
        if not np.allclose(np.asarray(holmfunc(p[perm])), a[perm]):
            return False
    return True


# ===================================================================================================== 1. exact permutation test
@pytest.mark.parametrize("alt", ALTS)
def test_theorem1_exact_type_I_validity_on_full_orbit(alt):
    assert type1_exact_ok(rv.permutation_pvalue, alt)


@pytest.mark.parametrize("nx,data", [(3, np.array([0.5, -1.2, 2.0, 0.1, 1.4, -0.7])), (2, np.array([3.0, -1.0, 0.2, 0.9, 1.6]))])
def test_theorem1_holds_for_unequal_arms_and_other_data(nx, data):
    for alt in ALTS:
        assert type1_exact_ok(rv.permutation_pvalue, alt, nx=nx, data=data)


@pytest.mark.parametrize("mutant", [mut_strict_inequality, mut_peek_direction])
def test_mutants_that_break_validity_are_caught(mutant):
    assert not type1_exact_ok(mutant, "two-sided")


def test_theorem2_monte_carlo_p_is_never_zero_and_close_to_exact():
    assert mc_pvalue_positive_ok(rv.permutation_pvalue)
    assert not mc_pvalue_positive_ok(mut_mc_no_plus_one)
    rng = np.random.default_rng(3)
    x, y = rng.normal(0.8, 1, 9), rng.normal(0, 1, 9)              # C(18,9)=48620 -> exact is feasible; compare with MC
    exact = rv.permutation_pvalue(x, y)
    m = 20000
    mc = rv.permutation_pvalue(x, y, max_exact=0, n_perm=m, seed=5)
    assert abs(mc - exact) <= 4 * np.sqrt(exact * (1 - exact) / m) + 1 / (m + 1)   # binomial 4-sigma envelope


def test_exact_identities_between_the_tails():
    """Exact algebra of the permutation distribution: P(T<=t) + P(T>=t) = 1 + P(T=t) >= 1;  for equal arms the distribution of T is symmetric,
    hence  p_two-sided = min(1, 2 min(p_less, p_greater))  (so the two-sided test can never be used as a disguised one-sided test)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        x, y = rng.normal(0.5, 1, 4), rng.normal(0, 1, 4)
        pl, pg, p2 = (rv.permutation_pvalue(x, y, a) for a in ("less", "greater", "two-sided"))
        assert pl + pg >= 1 - 1e-12
        assert abs(p2 - min(1.0, 2 * min(pl, pg))) < 1e-12


def test_symmetries_and_invariances():
    rng = np.random.default_rng(2)
    x, y = rng.normal(1, 1, 6), rng.normal(0, 1, 6)
    assert rv.permutation_pvalue(x, y) == pytest.approx(rv.permutation_pvalue(y, x))                       # label swap
    assert rv.permutation_pvalue(x, y, "less") == pytest.approx(rv.permutation_pvalue(y, x, "greater"))
    for a, b in ((2.5, -7.0), (0.01, 3.0)):                                                                 # positive affine maps
        for alt in ALTS:
            assert rv.permutation_pvalue(a * x + b, a * y + b, alt) == pytest.approx(rv.permutation_pvalue(x, y, alt))
    assert rv.permutation_pvalue(-x, -y, "less") == pytest.approx(rv.permutation_pvalue(x, y, "greater"))  # negation flips the tail


def test_global_random_state_is_never_touched_and_results_are_reproducible():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=12), rng.normal(size=12)
    a = rv.permutation_pvalue(x, y, max_exact=0, n_perm=3000, seed=9)
    b = rv.permutation_pvalue(x, y, max_exact=0, n_perm=3000, seed=9)
    rv.bootstrap_ci(x, n_boot=200, seed=1)
    assert a == b                                                   # same seed -> same p
    assert np.array_equal(before, np.random.get_state()[1])         # hidden global RNG untouched (no seed-fishing through shared state)
    assert rv.permutation_pvalue(x, y, max_exact=0, n_perm=3000, seed=10) != a


# ===================================================================================================== 3. Holm / multiplicity
def test_theorem3_holm_controls_fwer_exactly_independent():
    assert holm_fwer_ok(rv.holm, m=2, K=200, alpha=0.05, dependence="independent")   # exact: 1-(1-.025)^2 = .0494 <= .05
    assert holm_fwer_ok(rv.holm, m=3, K=40, alpha=0.15, dependence="independent")    # exact: 1-(1-.05)^3 = .1426 <= .15


@pytest.mark.parametrize("dep", ["comonotone", "disjoint"])
def test_theorem3_holm_controls_fwer_under_arbitrary_dependence(dep):
    assert holm_fwer_ok(rv.holm, m=2, K=200, alpha=0.05, dependence=dep)


@pytest.mark.parametrize("mutant", [mut_holm_none, mut_holm_off_by_one])
def test_uncorrected_mutants_are_caught_by_fwer_theorem(mutant):
    assert not holm_fwer_ok(mutant, m=2, K=200, alpha=0.05, dependence="independent")
    assert not holm_fwer_ok(mutant, m=2, K=200, alpha=0.05, dependence="disjoint")


def test_holm_algebra_and_known_values():
    assert holm_structure_ok(rv.holm)
    assert not holm_structure_ok(mut_holm_no_monotone)
    assert not holm_structure_ok(mut_holm_none)          # caught only because of the min(adj)=m*p_min identity
    assert not holm_structure_ok(mut_holm_off_by_one)
    assert np.allclose(rv.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(rv.holm([0.5]), [0.5])
    assert np.all(rv.holm([0.2, 0.9]) <= rv.bonferroni([0.2, 0.9]) + 1e-12)


def test_end_to_end_fwer_of_the_actual_pipeline_under_the_global_null():
    """Exact two-sided permutation test (5 v 5 seeds) on 4 contrasts + Holm, all nulls true: FWER must be <= alpha.
    Monte-Carlo tolerance = 3 binomial standard errors."""
    rng = np.random.default_rng(11)
    reps, alpha, hits = 800, 0.05, 0
    for _ in range(reps):
        p = [rv.permutation_pvalue(rng.normal(size=5), rng.normal(size=5)) for _ in range(4)]
        hits += rv.holm(p).min() <= alpha
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert hits / reps <= alpha + 3 * se


# ===================================================================================================== 4. minimum attainable p
@pytest.mark.parametrize("nx,ny", [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (2, 5)])
def test_min_attainable_p_equals_brute_force_minimum_over_all_assignments(nx, ny):
    """The infimum of p over data sets.  Equal arms: the mirror labelling always ties in |T| (T(x') = -T(x)), so exactly 2/C.
    Unequal arms: for ASYMMETRIC data (here 2^i, so the mirror is strictly less extreme) only the observed labelling attains it: 1/C."""
    n = nx + ny
    ranks = 2.0 ** np.arange(n)
    for alt in ("two-sided", "greater"):
        best = min(rv.permutation_pvalue(ranks[list(c)], np.delete(ranks, list(c)), alt) for c in itertools.combinations(range(n), nx))
        assert best == pytest.approx(rv.min_attainable_p(nx, ny, alt))


def test_three_seeds_can_never_reach_significance_five_seeds_can():
    assert rv.min_attainable_p(3, 3) == pytest.approx(0.1)               # > 0.05: impossible whatever the data
    assert 6 * rv.min_attainable_p(3, 3) > 0.05
    assert rv.min_attainable_p(5, 5) == pytest.approx(2 / 252)
    assert 4 * rv.min_attainable_p(5, 5) <= 0.05                          # a Holm family of 4 is still passable at n=5
    x, y = np.arange(3.0) + 10, np.arange(3.0)
    assert rv.permutation_pvalue(x, y) == pytest.approx(0.1)              # perfect separation, 3 v 3: p = 0.1 exactly


def test_decision_rule_refuses_to_call_impossible_or_null_results_negative():
    assert rv.decide(0.1, 0.05, min_p_adj_possible=0.1) == "UNDERPOWERED"
    assert rv.decide(0.01, 0.05, min_p_adj_possible=0.008) == "REJECT"
    assert rv.decide(0.4, 0.05, min_p_adj_possible=0.008) == "INCONCLUSIVE"       # there is no 'no effect' verdict


# ===================================================================================================== 5. optional stopping
def _welch_p_paths(reps, nmax, seed):
    rng = np.random.default_rng(seed)
    X, Y = rng.standard_normal((reps, nmax)), rng.standard_normal((reps, nmax))
    n = np.arange(1, nmax + 1)
    def stats_of(A):
        c, c2 = np.cumsum(A, 1), np.cumsum(A ** 2, 1)
        m = c / n
        return m, np.maximum((c2 - n * m ** 2) / np.maximum(n - 1, 1), 1e-12)
    mx, vx = stats_of(X)
    my, vy = stats_of(Y)
    se2 = (vx + vy) / n
    t = (mx - my) / np.sqrt(se2)
    df = se2 ** 2 / ((vx / n) ** 2 / np.maximum(n - 1, 1) * 1 + (vy / n) ** 2 / np.maximum(n - 1, 1))
    return 2 * stats.t.sf(np.abs(t), df)                                  # (reps, nmax), column j is the test at n=j+1


def test_theorem5_peeking_inflates_alpha_fixed_design_and_alpha_spending_do_not():
    """Armitage-McPherson-Rowe (1969): testing after every new pair of runs and stopping at the first p<.05 has a type-I error far above .05.
    Fixed n, or alpha split across K pre-planned looks, restores control.  Tolerances: 4 binomial standard errors."""
    reps, alpha, nmax = 4000, 0.05, 40
    P = _welch_p_paths(reps, nmax, seed=4)
    naive = np.mean((P[:, 4:] < alpha).any(1))                             # look at every n = 5..40
    fixed = np.mean(P[:, nmax - 1] < alpha)                                 # one pre-planned look at n = 40
    looks = np.arange(5, nmax + 1, 5) - 1                                   # K = 8 pre-planned looks
    spent = np.mean((P[:, looks] < rv.alpha_per_look(alpha, len(looks))).any(1))
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert naive > 0.15                                                     # the disease: > 3x the nominal level
    assert abs(fixed - alpha) <= 4 * se                                     # fixed design: at nominal level
    assert spent <= alpha + 4 * se                                          # alpha spending: controlled


def test_fixed_design_guard_refuses_any_other_sample_size():
    d = rv.FixedDesign(5)
    d.check(np.zeros(5), np.zeros(5))
    for bad in (4, 6, 50):
        with pytest.raises(rv.PreregistrationViolation):
            d.check(np.zeros(5), np.zeros(bad))


# ===================================================================================================== 7. bootstrap coverage
def test_bootstrap_ci_is_refused_below_n5_and_the_refusal_is_justified_by_coverage():
    with pytest.raises(rv.UnderpoweredError):
        rv.bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(6)
    reps = 1500

    def coverage(n):
        hit = 0
        for _ in range(reps):
            _, lo, hi = rv._percentile_bootstrap(rng.normal(0, 1, n), 0.95, 300, int(rng.integers(1 << 30)))
            hit += lo <= 0 <= hi
        return hit / reps

    c3, c30 = coverage(3), coverage(30)
    se = np.sqrt(0.95 * 0.05 / reps)
    assert c3 < 0.85                                       # n=3: nominal 95% interval covers far less  -> refusal is justified
    assert 0.925 - 2 * se <= c30 <= 0.965 + 4 * se         # n=30: close to nominal (percentile bootstrap is slightly liberal)


def test_bootstrap_ci_is_a_valid_interval_and_deterministic():
    x = np.array([1.0, 2.0, 2.5, 3.0, 4.5, 5.0])
    m, lo, hi = rv.bootstrap_ci(x, seed=1)
    assert lo <= m <= hi and (m, lo, hi) == rv.bootstrap_ci(x, seed=1)
    assert x.min() <= lo and hi <= x.max()                # a bootstrap mean cannot leave the data range


def test_min_n_cannot_be_overridden_through_the_public_api():
    with pytest.raises(TypeError):
        rv.bootstrap_ci([1.0, 2.0, 3.0], min_n=2)


# ===================================================================================================== 8. equivalence
def test_absence_of_significance_is_not_evidence_of_equality():
    """With few runs an equivalence claim is impossible (power ~ 0); with many runs and a wide margin it is nearly certain.
    So a non-significant difference test must be reported as INCONCLUSIVE, never as 'no difference'."""
    rng = np.random.default_rng(7)
    small = np.mean([rv.tost_equivalence(rng.normal(size=5), rng.normal(size=5), margin=0.2)["equivalent"] for _ in range(1500)])
    large = np.mean([rv.tost_equivalence(rng.normal(size=400), rng.normal(size=400), margin=0.5)["equivalent"] for _ in range(300)])
    assert small == 0.0
    assert large > 0.99
    with pytest.raises(rv.ValidationError):
        rv.tost_equivalence([1, 2, 3], [1, 2, 3.5], margin=0.0)                 # margin must be positive (and pre-registered)


def test_tost_type_I_error_at_the_margin_boundary_is_at_most_alpha():
    """At |true diff| = margin the equivalence claim is a FALSE positive; its rate must be <= alpha (size property of TOST)."""
    rng = np.random.default_rng(8)
    reps, n, margin, alpha = 3000, 60, 0.5, 0.05
    fp = np.mean([rv.tost_equivalence(rng.normal(margin, 1, n), rng.normal(0, 1, n), margin, alpha)["equivalent"] for _ in range(reps)])
    assert fp <= alpha + 4 * np.sqrt(alpha * (1 - alpha) / reps)


# ===================================================================================================== 9. sanitation
@pytest.mark.parametrize("bad", [[1, 2, np.nan], [1, 2, np.inf], [1.0], [[1, 2], [3, 4]]])
def test_corrupt_data_is_refused_not_silently_dropped(bad):
    with pytest.raises(rv.InvalidDataError):
        rv.permutation_pvalue(bad, [1, 2, 3, 4])
    with pytest.raises(rv.InvalidDataError):
        rv.permutation_pvalue([1, 2, 3, 4], bad)


@pytest.mark.parametrize("bad", [[-0.1, 0.5], [0.5, 1.2], [np.nan], []])
def test_invalid_p_values_are_refused_by_multiplicity_corrections(bad):
    with pytest.raises(rv.InvalidDataError):
        rv.holm(bad)
    with pytest.raises(rv.InvalidDataError):
        rv.bonferroni(bad)


def test_duplicate_seeds_are_refused_as_pseudo_replication():
    rv.check_unique_seeds([0, 1, 2, 3, 4])
    with pytest.raises(rv.InvalidDataError):
        rv.check_unique_seeds([0, 1, 2, 2, 4])


def test_unknown_alternative_is_refused():
    with pytest.raises(rv.ValidationError):
        rv.permutation_pvalue([1, 2, 3], [4, 5, 6], alternative="whichever-looks-better")


# ===================================================================================================== 6. pre-registration lock
def make_prereg(n=5):
    return rv.default_prereg(envs=["E"], pairs=[("a", "b"), ("a", "c")], metrics=["m1", "m2"], n_per_arm=n)


def test_lock_detects_every_single_field_tampering():
    pre = make_prereg()
    d0 = rv.digest(pre)
    rng = np.random.default_rng(0)
    digests = {d0}
    for i in range(300):
        q = copy.deepcopy(pre)
        kind = i % 6
        if kind == 0:
            q["alpha"] = float(rng.uniform(0.01, 0.2))
        elif kind == 1:
            q["min_seeds"] = int(rng.integers(1, 5))
        elif kind == 2:
            q["comparisons"][int(rng.integers(len(q["comparisons"])))]["alternative"] = "less"
        elif kind == 3:
            q["comparisons"].pop(int(rng.integers(len(q["comparisons"]))))                       # silently dropping a comparison
        elif kind == 4:
            q["n_per_arm"] = int(rng.integers(6, 30))
        else:
            q["comparisons"].append(dict(id=f"x{i}", family="E|m1", env="E", metric="m1", a="a", b="z", alternative="greater"))
        dq = rv.digest(q)
        assert dq != d0                                        # any change moves the SHA-256
        digests.add(dq)
        with pytest.raises(rv.PreregistrationViolation):
            rv.LockedAnalysis(q, d0)                           # and the analysis refuses to start
    assert len(digests) > 100                                  # no accidental collisions among distinct configs


def test_digest_is_canonical_key_order_invariant_and_rejects_nan():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert rv.digest(a) == rv.digest(b)
    with pytest.raises(ValueError):
        rv.digest({"alpha": float("nan")})                    # NaN would make the hash undefined


def _perfect(n=5):
    return np.arange(n) + 10.0, np.arange(n) + 0.0


def _run_all(la, n=5, effect=True):
    rng = np.random.default_rng(0)
    for cid in la.registry:
        x, y = _perfect(n) if effect else (rng.normal(size=n), rng.normal(size=n))
        la.run(cid, x, y)


def test_locked_analysis_only_runs_registered_comparisons_once_with_registered_n():
    pre = make_prereg()
    la = rv.LockedAnalysis(pre, rv.digest(pre))
    with pytest.raises(rv.PreregistrationViolation):
        la.run("E|m1|a|zzz", *_perfect())                     # unregistered comparison (forking paths)
    cid = next(iter(la.registry))
    with pytest.raises(rv.PreregistrationViolation):
        la.run(cid, *_perfect(6))                              # extra runs / optional stopping
    with pytest.raises(rv.PreregistrationViolation):
        la.run(cid, *_perfect(4))                              # dropped runs
    la.run(cid, *_perfect())
    with pytest.raises(rv.PreregistrationViolation):
        la.run(cid, *_perfect())                               # re-analysis of the same comparison


def test_results_are_sealed_until_the_whole_family_set_is_complete():
    pre = make_prereg()
    la = rv.LockedAnalysis(pre, rv.digest(pre))
    ids = list(la.registry)
    la.run(ids[0], *_perfect())
    with pytest.raises(rv.PreregistrationViolation):
        la.report()                                            # cannot peek at one result
    with pytest.raises(rv.PreregistrationViolation):
        la.finalize()                                          # cannot report only the comparisons that worked
    for i in ids[1:]:
        la.run(i, *_perfect())
    la.finalize()
    la.report()
    with pytest.raises(rv.PreregistrationViolation):
        la.run(ids[0], *_perfect())                            # sealed for good


def test_holm_is_applied_over_the_whole_registered_family_so_omitting_tests_cannot_help():
    pre = make_prereg()
    la = rv.LockedAnalysis(pre, rv.digest(pre))
    _run_all(la, effect=True)
    rep = la.finalize()
    fam = [i for i in la.registry if la.registry[i]["family"] == "E|m1"]      # 2 comparisons in this family
    p_raw = [rep[i]["p"] for i in fam]
    assert all(rep[i]["p_holm"] == pytest.approx(min(1.0, 2 * rep[i]["p"])) for i in fam)     # equal p -> Bonferroni-like factor m=2, not 1
    assert all(rep[i]["decision"] == "REJECT" for i in fam) and p_raw[0] == pytest.approx(2 / 252)


def test_underpowered_design_can_never_report_a_rejection_even_with_perfect_separation():
    pre = make_prereg(n=3)
    la = rv.LockedAnalysis(pre, rv.digest(pre))
    _run_all(la, n=3, effect=True)
    rep = la.finalize()
    assert all(v["decision"] == "UNDERPOWERED" and v["p"] == pytest.approx(0.1) for v in rep.values())


def test_null_data_is_inconclusive_not_negative():
    pre = make_prereg()
    la = rv.LockedAnalysis(pre, rv.digest(pre))
    _run_all(la, effect=False)
    assert all(v["decision"] in ("INCONCLUSIVE", "REJECT") and v["decision"] != "NO_EFFECT" for v in la.finalize().values())


def test_lock_files_round_trip_and_detect_editing_the_json_after_locking(tmp_path):
    jp, hp = str(tmp_path / "prereg.json"), str(tmp_path / "PREREG.sha256")
    d = rv.write_lock(make_prereg(), jp, hp)
    la = rv.load_locked(jp, hp)                                # honest load works
    assert len(la.registry) == 4                               # 1 env x 2 metrics x 2 pairs
    doc = json.load(open(jp))
    doc["alpha"] = 0.20                                        # the tamper: loosen alpha AFTER seeing results
    json.dump(doc, open(jp, "w"))
    with pytest.raises(rv.PreregistrationViolation):
        rv.load_locked(jp, hp)
    with pytest.raises(rv.PreregistrationViolation):
        rv.load_locked(str(tmp_path / "missing.json"), hp)     # no lock -> no confirmatory analysis
    assert d == rv.digest(make_prereg())


# ---- mutation tests for the lock itself: the guards must be the reason these tests pass
class LockedNoDesignCheck(rv.LockedAnalysis):
    def run(self, comparison_id, x, y):                       # MUTANT: skips the fixed-design guard
        c = self.registry[comparison_id]
        self._raw[comparison_id] = dict(p=rv.permutation_pvalue(x, y, c["alternative"]), diff=0.0, n=len(x), alternative=c["alternative"])


class LockedAllowsReanalysis(rv.LockedAnalysis):
    def run(self, comparison_id, x, y):                       # MUTANT: allows silent re-analysis (re-run until it works)
        self._raw.pop(comparison_id, None)
        super().run(comparison_id, x, y)


def _design_guard_holds(cls):
    pre = make_prereg()
    la = cls(pre, rv.digest(pre))
    try:
        la.run(next(iter(la.registry)), *_perfect(9))
    except rv.PreregistrationViolation:
        return True
    return False


def _reanalysis_guard_holds(cls):
    pre = make_prereg()
    la = cls(pre, rv.digest(pre))
    cid = next(iter(la.registry))
    la.run(cid, *_perfect())
    try:
        la.run(cid, *_perfect())
    except rv.PreregistrationViolation:
        return True
    return False


def test_lock_mutants_are_caught():
    assert _design_guard_holds(rv.LockedAnalysis) and not _design_guard_holds(LockedNoDesignCheck)
    assert _reanalysis_guard_holds(rv.LockedAnalysis) and not _reanalysis_guard_holds(LockedAllowsReanalysis)
