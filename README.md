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

## Statistical integrity (`evaluation/rigorous_validation.py`)
Every p-value, correction and confidence interval in this project is routed through one validated module, not scattered ad hoc
code. It exists because it is easy — often unconsciously — to bend an evaluation until p < 0.05 (the replication crisis): choosing
a tail after seeing the data, peeking and stopping early, reporting only the comparisons that worked, or widening a margin after
the fact. Each guard below is a theorem, and `tests/test_rigorous_validation.py` (52 tests) checks it by exact enumeration or by
simulation with a tolerance derived from the theory, then runs the same checker against deliberately broken "mutant" implementations
that must fail. `python -m beliefdelay.evaluation.rigorous_validation` prints the module docstring; `... lock` writes a hash-locked
`prereg.json`/`PREREG.sha256` pair; `analysis.py --locked` runs only the registered comparisons, once each, and refuses if the file
was edited afterward, if a family is incomplete, or if the seed count does not match what was locked in.

| Guard | Theorem | Failure it prevents |
|---|---|---|
| Exact permutation test | `P(p<=a)<=a` for every `a` (finite exchangeability) | inflated significance from a bespoke test statistic |
| Monte-Carlo p-value | `p=(b+1)/(m+1)`, never 0 (Phipson & Smyth 2010) | reporting `p=0` |
| Holm step-down | strong FWER control under *arbitrary* dependence (Holm 1979) | multiple-comparison fishing |
| Minimum attainable p | exact combinatorics; "cannot reach 0.05" is reported, not hidden | claiming significance a design could never show |
| Fixed design / alpha-spending | peeking inflates alpha (Armitage–McPherson–Rowe 1969) | optional stopping |
| Pre-registration lock | SHA-256 over canonical JSON; run-once, whole-family, sealed-until-final | changing the hypothesis after seeing results |
| Bootstrap CI | refused below n=5 (coverage collapses; shown by simulation) | a false sense of precision from 3 seeds |
| Equivalence (TOST) | a non-significant difference is INCONCLUSIVE, never "no effect" | overclaiming a null result |
| Input sanitation | NaN/inf/duplicate seeds are refused, not silently dropped | quietly discarding inconvenient runs |

## Making "it learned automatically" explicit (`explicitness.py`)
Composes existing instruments into four independently falsifiable axes per model: SUFFICIENCY (excess KL/skill),
DECODABILITY (PSR/belief probes), DYNAMICAL FORM (does h_t evolve linearly, with a spectrum matching the process's
own -- `env.koopman_eig()`?), STABILITY (amplification growth under a small parameter perturbation -- Luo et al. 2024,
NeurIPS, "Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate", arXiv:2405.15384).
Calibrated against an oracle (must score near ceiling) and a blind model (must score near floor) before being
trusted on real checkpoints; building that calibration caught two real bugs (see THEORY.md §11). Run on any saved
checkpoint: `python -m beliefdelay.explicitness results/cpu --env grid_patrol`.

## Split learning rate (`--split-lr`)
Luo et al. 2024 prove that for any recurrence with contractive hidden dynamics (K_h<1 -- satisfied by GRU/LSTM via
sigmoid gates and by Mamba/RWKV via their bounded per-channel decay), a single gradient step's effect on the output
is amplified across rollout length and converges to a fixed factor. In our models the recurrent core is ~99% of the
parameters, so training the whole network at one learning rate is exactly the confound they describe. `--split-lr`
runs a two-stage search (tuning.py) for a separate core vs. head learning rate; `amplification.py` reproduces their
diagnostic directly (validated against an exact toy linear RNN in tests/test_amplification.py) and should be run on
TRAINED checkpoints, not fresh ones -- at random init none of our four architectures show real amplification, since
the gates/decays have not yet learned to hold long memory.

## Status
No hypothesis test has been run. The first CPU run (abstract envs, 3 seeds) is disclosed in the pre-registration and motivated the fixes above; its LR search was clipped at the top of the grid in 7 of 12 cells, so its architecture ranking should not be trusted.
