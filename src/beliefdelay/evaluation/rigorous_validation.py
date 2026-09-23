"""Rigorous validation layer: the statistical tools that judge our experiments, built so they cannot be quietly misused.

Design rule: a result is only as trustworthy as the procedure that produced its p-value.  Each guard below is the programmed form of
a theorem (see tests/test_rigorous_validation.py, where every guard is verified by exact enumeration or by a simulation whose
tolerance is derived from the theory, and where deliberately WRONG "mutant" implementations must be caught).

  1. Exact permutation test            P(p <= a) <= a for every a  (finite group argument; exact enumeration)
  2. Monte-Carlo p-values              (b+1)/(m+1): valid and never 0  (Phipson & Smyth 2010)
  3. Holm step-down                    strong FWER control under ARBITRARY dependence (Holm 1979)
  4. Minimum attainable p              exact combinatorics: 2/C(n,k) (equal arms) or 1/C(n,k); "cannot be significant" is refused, not hidden
  5. Fixed design (no optional stopping)  peeking inflates alpha (Armitage-McPherson-Rowe 1969): analysis refuses n != planned n
  6. Pre-registration lock             SHA-256 of canonical JSON; only registered comparisons; each analysed once; families complete;
                                       results sealed until finalize(); direction fixed in advance
  7. Bootstrap CI                      refused for n < 5 (percentile bootstrap under-covers; proved by simulation in the tests)
  8. Equivalence (TOST)                "no difference" may only be claimed via equivalence with a pre-registered margin;
                                       a non-significant difference test is INCONCLUSIVE, never "no effect"
  9. Input sanitation                  NaN/inf, duplicates, wrong shapes: failures must be recorded at the pre-registered worst value, not dropped
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

MAX_EXACT = 200_000
MIN_BOOT_N = 5          # a constant, deliberately NOT a parameter of bootstrap_ci: it cannot be overridden by a caller


class ValidationError(Exception):
    """Base class: the requested statistical operation is invalid."""


class PreregistrationViolation(ValidationError):
    """The analysis deviates from the locked pre-registration."""


class UnderpoweredError(ValidationError):
    """The sample is too small for the requested inference to be valid."""


class InvalidDataError(ValidationError):
    """Data would silently corrupt the statistic (NaN/inf, wrong shape, too few points)."""


# ------------------------------------------------------------------------------------------------------ input sanitation
def clean(x, name: str = "x", min_n: int = 2) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise InvalidDataError(f"{name}: must be 1-D")
    if a.size < min_n:
        raise InvalidDataError(f"{name}: need >= {min_n} observations, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise InvalidDataError(f"{name}: contains NaN/inf. Failed runs must be recorded at the pre-registered worst value, not dropped.")
    return a


def check_unique_seeds(seeds) -> None:
    """Pseudo-replication guard: repeated seeds are not independent replicates."""
    s = list(seeds)
    if len(set(s)) != len(s):
        raise InvalidDataError("duplicate seeds: repeated runs are not independent replicates")


@lru_cache(maxsize=32)
def _combo_index(n: int, k: int) -> np.ndarray:
    return np.fromiter((i for c in combinations(range(n), k) for i in c), dtype=np.int64).reshape(-1, k)


# ------------------------------------------------------------------------------------------------------ 1-2. permutation test
def permutation_pvalue(x, y, alternative: str = "two-sided", n_perm: int = 20000, seed: int = 0, max_exact: int = MAX_EXACT) -> float:
    """Permutation test of the difference of means T = mean(x) - mean(y) under exchangeability.

    Exact (a fraction of all C(n, |x|) label assignments) when feasible, otherwise Monte-Carlo with the (b+1)/(m+1) correction.
    alternative: 'two-sided' (|T|), 'less' (T small), 'greater' (T large).  Uses a private RNG: global random state is never touched.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValidationError(f"unknown alternative {alternative!r}")
    x, y = clean(x, "x"), clean(y, "y")
    pooled, nx = np.concatenate([x, y]), len(x)
    n, total = len(pooled), pooled.sum()
    obs = x.mean() - y.mean()

    def count(T):
        tol = 1e-12 * max(1.0, abs(obs), float(np.abs(T).max()))
        if alternative == "less":
            return int((T <= obs + tol).sum())
        if alternative == "greater":
            return int((T >= obs - tol).sum())
        return int((np.abs(T) >= abs(obs) - tol).sum())

    if comb(n, nx) <= max_exact:
        idx = _combo_index(n, nx)
        sx = pooled[idx].sum(1)
        T = sx / nx - (total - sx) / (n - nx)
        return count(T) / len(T)
    rng = np.random.default_rng(seed)
    b, done, chunk = 0, 0, 2000
    while done < n_perm:
        m = min(chunk, n_perm - done)
        order = np.argsort(rng.random((m, n)), axis=1)[:, :nx]
        sx = pooled[order].sum(1)
        b += count(sx / nx - (total - sx) / (n - nx))
        done += m
    return (b + 1) / (n_perm + 1)


def min_attainable_p(nx: int, ny: int, alternative: str = "two-sided") -> float:
    """Infimum of the permutation p-value over all data sets with these arm sizes.  Equal arms: the mirror labelling always ties, so 2/C;
    unequal arms (two-sided) or one-sided: 1/C.  Exact combinatorics, verified by brute force in the tests."""
    c = comb(nx + ny, nx)
    return 2.0 / c if (alternative == "two-sided" and nx == ny) else 1.0 / c


# ------------------------------------------------------------------------------------------------------ 3. multiplicity
def _check_p(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise InvalidDataError("p-values must be a non-empty 1-D array in [0, 1]")
    return p


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values: strong FWER control at any alpha under arbitrary dependence."""
    p = _check_p(p)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bonferroni(p) -> np.ndarray:
    p = _check_p(p)
    return np.minimum(1.0, len(p) * p)


def alpha_per_look(alpha: float, n_looks: int) -> float:
    """The only legitimate way to look at accumulating data: split alpha across the K pre-planned looks (Bonferroni spending)."""
    if n_looks < 1:
        raise ValidationError("n_looks >= 1")
    return alpha / n_looks


# ------------------------------------------------------------------------------------------------------ 7. bootstrap
def _percentile_bootstrap(x: np.ndarray, level: float, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_ci(x, level: float = 0.95, n_boot: int = 10000, seed: int = 0):
    """Percentile bootstrap CI for the mean.  Refused for n < MIN_BOOT_N (coverage collapses there; proved by simulation in the tests)."""
    x = clean(x, "x", min_n=2)
    if len(x) < MIN_BOOT_N:
        raise UnderpoweredError(f"bootstrap CI refused: n={len(x)} < {MIN_BOOT_N}; show the raw points instead")
    return _percentile_bootstrap(x, level, n_boot, seed)


# ------------------------------------------------------------------------------------------------------ 8. equivalence
def tost_equivalence(x, y, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided Welch tests: H0 |mean(x)-mean(y)| >= margin.  The margin must be fixed BEFORE seeing data."""
    from scipy import stats
    if not margin > 0:
        raise ValidationError("equivalence margin must be > 0 and pre-registered")
    x, y = clean(x, "x"), clean(y, "y")
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1) / nx, y.var(ddof=1) / ny
    se = float(np.sqrt(vx + vy))
    if se == 0:
        raise InvalidDataError("zero variance in both arms: equivalence test undefined")
    df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    d = x.mean() - y.mean()
    p = max(float(stats.t.sf((d + margin) / se, df)), float(stats.t.cdf((d - margin) / se, df)))
    return dict(p=p, equivalent=bool(p < alpha), diff=float(d), margin=float(margin))


def decide(p_adj: float, alpha: float, min_p_adj_possible: float) -> str:
    """UNDERPOWERED if no data could ever have rejected; REJECT; otherwise INCONCLUSIVE.  There is no 'no effect' outcome."""
    if min_p_adj_possible > alpha:
        return "UNDERPOWERED"
    return "REJECT" if p_adj <= alpha else "INCONCLUSIVE"


# ------------------------------------------------------------------------------------------------------ 5. fixed design
class FixedDesign:
    """Sample size fixed in advance.  Any other n is optional stopping or selective dropping and is refused."""

    def __init__(self, n_per_arm: int):
        self.n = int(n_per_arm)

    def check(self, *arms) -> None:
        for a in arms:
            if len(a) != self.n:
                raise PreregistrationViolation(f"fixed design: expected n={self.n} per arm, got {len(a)} (no peeking, no extra runs, no dropping)")


# ------------------------------------------------------------------------------------------------------ 6. pre-registration lock
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


class LockedAnalysis:
    """Confirmatory analysis bound to a hash-locked pre-registration.

    * construction fails if the pre-registration differs from the recorded digest (tamper evidence);
    * only registered comparisons can be run, each once, with the registered arm size, direction and family;
    * Holm is applied over the WHOLE registered family, so omitting comparisons cannot lower an adjusted p;
    * results are sealed until finalize(), which requires every registered comparison to have been run.
    """

    def __init__(self, prereg: dict, expected_digest: str, seed: int = 0):
        if digest(prereg) != expected_digest:
            raise PreregistrationViolation("pre-registration does not match its locked digest: it was modified after locking")
        self.prereg, self.alpha, self.seed = prereg, float(prereg["alpha"]), seed
        self.min_seeds = int(prereg["min_seeds"])
        self.design = FixedDesign(prereg["n_per_arm"])
        self.registry = {c["id"]: c for c in prereg["comparisons"]}
        if len(self.registry) != len(prereg["comparisons"]):
            raise PreregistrationViolation("duplicate comparison ids in pre-registration")
        self.families: dict[str, list[str]] = {}
        for c in prereg["comparisons"]:
            self.families.setdefault(c["family"], []).append(c["id"])
        self._raw: dict[str, dict] = {}
        self._final: dict[str, dict] | None = None

    def run(self, comparison_id: str, x, y) -> None:
        if self._final is not None:
            raise PreregistrationViolation("analysis already finalised")
        if comparison_id not in self.registry:
            raise PreregistrationViolation(f"unregistered comparison {comparison_id!r}: not in the pre-registration")
        if comparison_id in self._raw:
            raise PreregistrationViolation(f"comparison {comparison_id!r} was already analysed; re-analysis is forbidden")
        c = self.registry[comparison_id]
        x, y = clean(x, "x"), clean(y, "y")
        self.design.check(x, y)
        p = permutation_pvalue(x, y, alternative=c["alternative"], seed=self.seed)
        self._raw[comparison_id] = dict(p=p, diff=float(x.mean() - y.mean()), n=len(x), alternative=c["alternative"])

    def finalize(self) -> dict:
        missing = [i for i in self.registry if i not in self._raw]
        if missing:
            raise PreregistrationViolation(f"{len(missing)} registered comparisons were not run (e.g. {missing[0]!r}); partial families are refused")
        out: dict[str, dict] = {}
        for fam, ids in self.families.items():
            adj = holm([self._raw[i]["p"] for i in ids])
            for i, a in zip(ids, adj):
                r = self._raw[i]
                pmin = min(1.0, len(ids) * min_attainable_p(r["n"], r["n"], r["alternative"]))
                status = decide(float(a), self.alpha, pmin)
                if r["n"] < self.min_seeds:
                    status = "UNDERPOWERED"
                out[i] = dict(r, family=fam, p_holm=float(a), min_p_holm_possible=pmin, decision=status)
        self._final = out
        return out

    def report(self) -> dict:
        if self._final is None:
            raise PreregistrationViolation("results are sealed until finalize(): no peeking at individual comparisons")
        return self._final


def default_prereg(envs=None, pairs=None, metrics=None, n_per_arm: int = 5) -> dict:
    """Pre-registration v3 as data (see PREREGISTRATION.md): two-sided contrasts, Holm within each (env, metric) family."""
    envs = envs or ["grid_patrol", "grid_drift"]
    pairs = pairs or [("lstm", "rwkv"), ("lstm", "mamba"), ("transformer", "rwkv"), ("transformer", "mamba")]
    metrics = metrics or ["excess_kl", "probe_psr_score", "prof_w1"]
    comps = [dict(id=f"{e}|{m}|{a}|{b}", family=f"{e}|{m}", env=e, metric=m, a=a, b=b, alternative="two-sided")
             for e in envs for m in metrics for a, b in pairs]
    return dict(version="v3", alpha=0.05, min_seeds=5, n_per_arm=n_per_arm, comparisons=comps)


def write_lock(prereg: dict, json_path: str = "prereg.json", hash_path: str = "PREREG.sha256") -> str:
    with open(json_path, "w") as f:
        f.write(canonical_json(prereg))
    d = digest(prereg)
    with open(hash_path, "w") as f:
        f.write(d + "\n")
    return d


def load_locked(json_path: str = "prereg.json", hash_path: str = "PREREG.sha256", seed: int = 0) -> LockedAnalysis:
    if not (os.path.exists(json_path) and os.path.exists(hash_path)):
        raise PreregistrationViolation(f"no lock found ({json_path}, {hash_path}); run `python -m beliefdelay.evaluation.rigorous_validation lock`")
    return LockedAnalysis(json.load(open(json_path)), open(hash_path).read().strip(), seed=seed)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "lock":
        print("locked, sha256 =", write_lock(default_prereg()))
    else:
        print(__doc__)
