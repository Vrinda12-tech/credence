# beliefdelay

An agent explores a map looking for a **hidden object** with a noisy proximity sensor. The Bayes-optimal memory of that search is a **belief map**,
computed exactly. We train sequence models (LSTM, Transformer, RWKV, Mamba, plus a literal delay-embedding MLP as the classical null) on the agent's stream
and ask how the belief is stored, and whether the *kind of memory* (in-place real decay / dense recurrence / lag-addressable) decides how faithful it is.
Prediction is the training signal; the object of study is the internal belief, compared with the exact posterior (Bayes filter) or, in the oscillator, the exact Kalman filter.

Read in order: `THEORY.md` (proved vs motivated; classical grounding) -> `PREREGISTRATION.md` (v3, with full disclosure of data seen) -> code.

| env | hidden dynamics | exact belief | memory demand |
|---|---|---|---|
| `grid_static` | object never moves | additive accumulator of log-evidence (a selective *diagonal* recurrence) | integrate forever |
| `grid_drift`  | lazy random walk | evidence forgotten geometrically | exponential decay is exact (control) |
| `grid_patrol` | circles a 16-cell ring | lag-`k` evidence shifted by `k` (Prop 6) | rotate/shift stored evidence |
| `osc_rotating` / `osc_decay` | linear-Gaussian, complex / real spectrum | Kalman filter | damped-oscillation / monotone kernel (Prop 10) |
| `delayed_readout`, `random_fast/slow` | abstract POMDPs | Bayes filter | theory sanity checks only |

## Design principles (each one came from a specific flaw in an earlier run; see PREREGISTRATION.md)
1. **Never rank untuned models.** LR search extends past grid edges and defaults on a flat grid (`tuning.py`).
2. **Measure the right biomarker.** Primary probe = predictive-state (PSR) vector; raw belief map secondary; low-dimensional moments (expected position, entropy, ring circular moments) as R^2. Probe train scores are stored to expose over-fitting.
3. **Report skill, not raw error** (`1 - excess/KL(marginal)`), plus convergence and LR-bound flags per run.
4. **A test must be able to pass:** >=5 seeds, two-sided permutation tests, the minimum attainable p printed.
5. **Change one mathematical property at a time** (matched pairs: patrol vs drift; osc_rotating vs osc_decay) and test the *interaction*.
6. **Include the classical null** (`delay_mlp`: an MLP on exactly W=8 lags) and exact baselines (Bayes = 0, window filter eps(W)).

## Layout
```
src/beliefdelay/grid.py           grid world, exact belief map, window filter, PSR test vectors, belief moments, seeker policy
src/beliefdelay/lgssm.py          linear-Gaussian oscillators with exact Kalman filter and kernel
src/beliefdelay/pomdp.py          abstract POMDPs (sanity checks)
src/beliefdelay/models.py         LSTM, GPT-2, RWKV-4 (HF), Mamba (mambapy), delay_mlp, mamba_noconv; CPU-only
src/beliefdelay/metrics.py        excess KL + skill, PSR/belief/moment probes, lag-influence profile, L_eff
src/beliefdelay/tuning.py         edge-aware learning-rate search
src/beliefdelay/run_experiment.py grid/abstract runner (resumable, checkpoints);  run_lgssm.py: oscillator runner + analysis
src/beliefdelay/analysis.py       per-seed tables, two-sided tests, Holm, interaction statistic, flags, figures;  viz.py: belief maps
tests/                            35 tests: brute-force filters, Prop 6 shift identity, Kalman == Gaussian conditioning, LR search, probes, causality
```

## Quick start
```bash
pip install -e .                       # or: uv pip install -e .
pytest -q                              # must pass (35 tests, ~10 s)
python -m beliefdelay.run_experiment --preset smoke --out results/smoke      # plumbing only
# oscillator pair (cheap, cleanest test of Prop 7):
python -m beliefdelay.run_lgssm --preset cpu --out results/lgssm
python -m beliefdelay.run_lgssm --analyze results/lgssm --fig figures/kernels.png
# grid (hours on CPU; use the Colab notebook for --preset small/full):
python -m beliefdelay.run_experiment --preset cpu --archs lstm transformer rwkv mamba delay_mlp --out results/grid
python -m beliefdelay.analysis results/grid --out figures/grid
python -m beliefdelay.viz results/grid --env grid_patrol --out figures/maps_patrol.png
```
Presets: `smoke` (seconds), `cpu_quick` (the old 3-seed 1500-step run; exploratory only), `cpu` (5 seeds, 3000 steps), `small`/`full` (GPU). Runs resume; JSONs record LR, flags, `d_model`, `n_params`.

## Status
No hypothesis test has been run. The first CPU run (abstract envs, 3 seeds) is disclosed in the pre-registration and motivated the fixes above; its LR search was clipped at the top of the grid in 7 of 12 cells, so its architecture ranking should not be trusted.
