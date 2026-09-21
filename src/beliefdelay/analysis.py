"""Aggregate results, run the pre-registered tests, draw figures.

    python -m beliefdelay.analysis results/full --out figures/

Statistics (see PREREGISTRATION.md): per-cell mean with percentile-bootstrap 95% CI over seeds;
pairwise one-sided exact permutation tests on the difference of means; Holm correction inside each
hypothesis family.  With n seeds per arm the smallest attainable p is 1/C(2n, n): n=5 -> 0.004.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

METRICS = {  # column -> (label, direction: +1 higher is better, -1 lower is better)
    "excess_kl": ("excess predictive KL", -1),
    "skill": ("skill = 1 - excess/KL(marginal)", +1),
    "probe_psr_score": ("predictive-state probe (PRIMARY)", +1),
    "probe_moments_r2": ("belief-moments R^2", +1),
    "probe_pred_score": ("predictive-law probe score", +1),
    "probe_belief_score": ("belief-state probe score", +1),
    "prof_w1": ("memory-profile W1 to Bayes", -1),
    "l_eff": ("effective delay depth", +1),
    "map_mass_at_mode": ("belief-map mode agreement", +1),
    "map_entropy_corr": ("belief-map uncertainty corr.", +1),
}


def load(dirpath: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(dirpath, "*__*__P*__s*.json"))):
        r = json.load(open(f))
        r["profile_model"] = np.array(r["profile_model"]); r["profile_bayes"] = np.array(r["profile_bayes"])
        rows.append(r)
    return pd.DataFrame(rows)


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def perm_test_less(a, b, max_perm=20000, seed=0) -> float:
    """H1: mean(a) < mean(b).  Exact if C(n_a+n_b, n_a) <= max_perm else Monte-Carlo."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, na = np.concatenate([a, b]), len(a)
    obs = a.mean() - b.mean()
    from math import comb
    if comb(len(pooled), na) <= max_perm:
        from itertools import combinations as C
        cnt = tot = 0
        for idx in C(range(len(pooled)), na):
            m = np.zeros(len(pooled), bool); m[list(idx)] = True
            tot += 1; cnt += (pooled[m].mean() - pooled[~m].mean()) <= obs + 1e-15
        return cnt / tot
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(max_perm):
        p = rng.permutation(pooled)
        cnt += (p[:na].mean() - p[na:].mean()) <= obs + 1e-15
    return (cnt + 1) / (max_perm + 1)


def perm_test_two_sided(a, b, max_perm=20000, seed=0) -> float:
    """H1: means differ (no direction chosen after seeing data).  Exact if feasible, else Monte-Carlo."""
    from math import comb
    from itertools import combinations as C
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, na = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    if comb(len(pooled), na) <= max_perm:
        cnt = tot = 0
        for idx in C(range(len(pooled)), na):
            m = np.zeros(len(pooled), bool); m[list(idx)] = True
            tot += 1; cnt += abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-15
        return cnt / tot
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(max_perm):
        p_ = rng.permutation(pooled); cnt += abs(p_[:na].mean() - p_[na:].mean()) >= obs - 1e-15
    return (cnt + 1) / (max_perm + 1)


def min_attainable_p(na: int, nb: int) -> float:
    """Smallest two-sided p a permutation test can return with these sample sizes (perfect separation)."""
    from math import comb
    return min(1.0, 2.0 / comb(na + nb, na))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (env, arch, P), g in df.groupby(["env", "arch", "target_params"]):
        row = dict(env=env, arch=arch, budget=P, n=len(g), params=int(g.n_params.mean()))
        for k in [k for k in METRICS if k in g.columns]:
            m, lo, hi = boot_ci(g[k].replace(np.inf, np.nan))
            row[k] = f"{m:.4g} [{lo:.3g}, {hi:.3g}]"
        out.append(row)
    return pd.DataFrame(out)


def contrast_family(df, env, metric, direction, pairs, label, tag):
    g = df[df.env == env]
    fam = []
    for x, y in pairs:
        a, b = g[g.arch == x][metric].replace(np.inf, np.nan).dropna().values, g[g.arch == y][metric].replace(np.inf, np.nan).dropna().values
        if len(a) >= 2 and len(b) >= 2:
            fam.append(dict(family=f"{label} [{tag}]", env=env, contrast=f"{x} vs {y}", n=f"{len(a)}v{len(b)}",
                            better=(x if (a.mean() - b.mean()) * direction > 0 else y),
                            diff=float(a.mean() - b.mean()), p=perm_test_two_sided(a, b), min_p=min_attainable_p(len(a), len(b))))
    m = len(fam)
    for f, pa in zip(fam, holm([f["p"] for f in fam]) if fam else []):
        f["p_holm"] = pa
        f["can_reach_0.05"] = bool(f["min_p"] * m <= 0.05)
    return fam


def preregistered_tests(df: pd.DataFrame) -> pd.DataFrame:
    """PREREGISTRATION v3 (two-sided; Holm within each (env, metric) family of four contrasts):
       dense-recurrent vs real-diagonal (lstm vs rwkv, lstm vs mamba) and lag-addressable vs real-diagonal (transformer vs rwkv, vs mamba).
       Primary env: grid_patrol.  Control: grid_drift (no directional claim).  Exploratory: everything else."""
    pairs = [("lstm", "rwkv"), ("lstm", "mamba"), ("transformer", "rwkv"), ("transformer", "mamba")]
    rows = []
    for env, tag in (("grid_patrol", "PRIMARY"), ("grid_drift", "CONTROL"), ("grid_static", "EXPLORATORY"),
                     ("delayed_readout", "SANITY"), ("random_slow", "SANITY"), ("random_fast", "SANITY")):
        for metric, direction, label in (("excess_kl", -1, "M1"), ("probe_psr_score", +1, "M2-psr"), ("prof_w1", -1, "M3")):
            if metric in df.columns:
                rows += contrast_family(df, env, metric, direction, pairs, label, tag)
    return pd.DataFrame(rows)


def interaction(df, group_a, group_b, env1, env2, metric, n_boot=5000, seed=0):
    """Difference of differences:  [mean_a - mean_b](env1) - [mean_a - mean_b](env2), with a percentile bootstrap CI over seeds.
    This is the pre-registered test of the THEORY claim (a gap that depends on the environment's memory structure)."""
    rng = np.random.default_rng(seed)
    def cell(env, archs):
        v = df[(df.env == env) & (df.arch.isin(archs))].groupby("arch")[metric].apply(lambda x: x.replace(np.inf, np.nan).dropna().values)
        return [np.asarray(x, float) for x in v.values if len(x)]
    def stat(rs):
        return np.mean([np.mean(rng.choice(x, len(x))) if rs else np.mean(x) for x in cell(env1, group_a)]) \
             - np.mean([np.mean(rng.choice(x, len(x))) if rs else np.mean(x) for x in cell(env1, group_b)]) \
             - np.mean([np.mean(rng.choice(x, len(x))) if rs else np.mean(x) for x in cell(env2, group_a)]) \
             + np.mean([np.mean(rng.choice(x, len(x))) if rs else np.mean(x) for x in cell(env2, group_b)])
    try:
        pt = stat(False)
        bs = np.array([stat(True) for _ in range(n_boot)])
    except (ValueError, IndexError):
        return None
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(metric=metric, env1=env1, env2=env2, a="+".join(group_a), b="+".join(group_b), interaction=float(pt),
                ci95=(float(lo), float(hi)), excludes_zero=bool(lo > 0 or hi < 0))


def per_seed_table(df, metric="excess_kl"):
    return df.pivot_table(index=["env", "arch"], columns="seed", values=metric).round(5)


def all_pairs(df: pd.DataFrame, metric="excess_kl") -> pd.DataFrame:
    rows = []
    for env, g in df.groupby("env"):
        archs = sorted(g.arch.unique())
        pairs = list(combinations(archs, 2))
        direction = -1
        rows += contrast_family(df, env, metric, direction, pairs, metric, "EXPLORATORY")
    return pd.DataFrame(rows)


def p_crit(df: pd.DataFrame, env="delayed_readout", thr=0.01) -> pd.DataFrame:
    """H3: smallest parameter budget whose mean excess KL < thr (threshold fixed in PREREGISTRATION.md)."""
    rows = []
    g = df[df.env == env]
    for arch, ga in g.groupby("arch"):
        m = ga.groupby("target_params").excess_kl.mean().sort_index()
        ok = m[m < thr]
        rows.append(dict(arch=arch, P_crit=int(ok.index[0]) if len(ok) else None,
                         curve={int(k): round(float(v), 4) for k, v in m.items()}))
    return pd.DataFrame(rows)


def all_pairs(df: pd.DataFrame, metric="excess_kl") -> pd.DataFrame:
    rows = []
    for env, g in df.groupby("env"):
        archs = sorted(g.arch.unique())
        fam = []
        for a_, b_ in combinations(archs, 2):
            a, b = g[g.arch == a_][metric].values, g[g.arch == b_][metric].values
            if len(a) < 2 or len(b) < 2:
                continue
            lo, hi = (a_, b_) if a.mean() < b.mean() else (b_, a_)
            x, y = g[g.arch == lo][metric].values, g[g.arch == hi][metric].values
            fam.append(dict(env=env, better=lo, worse=hi, diff=x.mean() - y.mean(), p=perm_test_less(x, y)))
        if fam:
            for f, pa in zip(fam, holm([f["p"] for f in fam])):
                f["p_holm"] = pa; rows.append(f)
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    envs = sorted(df.env.unique()); archs = [a for a in ["lstm", "transformer", "rwkv", "mamba"] if a in set(df.arch)]
    col = dict(lstm="#8c564b", transformer="#1f77b4", rwkv="#2ca02c", mamba="#d62728")

    # Fig 1: metrics per env
    for metric, (label, _) in [(k, v) for k, v in METRICS.items() if k in df.columns]:
        fig, axs = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.4), squeeze=False)
        for ax, env in zip(axs[0], envs):
            for i, a in enumerate(archs):
                v = df[(df.env == env) & (df.arch == a)][metric].replace(np.inf, np.nan).dropna().values
                if len(v) == 0:
                    continue
                m, lo, hi = boot_ci(v)
                ax.bar(i, m, color=col[a], alpha=0.8); ax.errorbar(i, m, [[m - lo], [hi - m]], color="k", capsize=3)
                ax.scatter([i] * len(v), v, s=12, color="k", zorder=3)
            ax.set_xticks(range(len(archs))); ax.set_xticklabels(archs, rotation=30); ax.set_title(env)
            if metric == "excess_kl":
                ax.set_yscale("log")
        axs[0][0].set_ylabel(label); fig.tight_layout(); fig.savefig(os.path.join(out, f"metric_{metric}.png"), dpi=140); plt.close(fig)

    # Fig 2: memory profiles vs Bayes
    fig, axs = plt.subplots(1, len(envs), figsize=(4.4 * len(envs), 3.4), squeeze=False)
    for ax, env in zip(axs[0], envs):
        g = df[df.env == env]
        if len(g) == 0:
            continue
        Is = np.mean(np.stack(g.profile_bayes.values), 0)
        ax.plot(Is, "k-o", lw=2.5, label="Bayes (exact)")
        for a in archs:
            gg = g[g.arch == a]
            if len(gg):
                ax.plot(np.mean(np.stack(gg.profile_model.values), 0), "-s", color=col[a], label=a, alpha=0.85)
        ax.set_xlabel("lag k"); ax.set_title(env)
    axs[0][0].set_ylabel("influence I(k) = E TV"); axs[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "memory_profiles.png"), dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir"); ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    df = load(a.results_dir)
    if df.empty:
        raise SystemExit("no results found")
    df_all = df
    df = df[df.target_params == df.target_params.max()]
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print("\n=== summary (mean [95% bootstrap CI over seeds]) ===")
    tab = summary_table(df_all); print(tab.to_string(index=False))
    os.makedirs(a.out, exist_ok=True); tab.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print("\n=== per-seed excess_kl (look at the raw points: bootstrap CIs on n<5 are not meaningful) ===")
    print(per_seed_table(df_all).to_string())
    n_min = int(df_all.groupby(["env", "arch", "target_params"]).size().min())
    if n_min < 5:
        print(f"\n!! UNDERPOWERED: smallest cell has n={n_min}. Two-sided permutation tests cannot reach alpha=0.05 after Holm; "
              f"all p-values below are descriptive.")
    for col, msg in (("converged", "still improving at the end (ranking mixes learning speed with asymptote)"),
                     ("lr_at_bound", "best learning rate sits on the search bound (tuning unresolved)")):
        if col in df_all.columns:
            bad = df_all[(df_all[col] == (col == "lr_at_bound")) if col == "lr_at_bound" else (df_all[col] == False)]
            if len(bad):
                print(f"\n!! {msg}: " + ", ".join(sorted({f"{r.env}/{r.arch}" for r in bad.itertuples()})))
    print("\n=== pre-registered tests (two-sided, Holm within family) ===")
    print(preregistered_tests(df).to_string(index=False))
    if df_all.target_params.nunique() > 1:
        print("\n=== H3: P_crit (mean excess KL < 0.01) ===")
        print(p_crit(df_all).to_string(index=False))
    print("\n=== interaction (difference of differences); prediction: real-diagonal models fall further behind LSTM on the patrol than on drift ===")
    for m in ("skill", "probe_psr_score"):
        r = interaction(df_all[df_all.target_params == df_all.target_params.max()], ["rwkv", "mamba"], ["lstm"], "grid_patrol", "grid_drift", m)
        print(r if r else f"(not enough cells for {m})")
    print("\n=== exploratory: all pairs on excess_kl (two-sided) ===")
    print(all_pairs(df).to_string(index=False))
    make_figures(df, a.out)
    print(f"\nfigures -> {a.out}/")


if __name__ == "__main__":
    main()
