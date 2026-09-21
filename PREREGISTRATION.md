# PREREGISTRATION.md  (v3)

**Commit this file to git *before* running any `small`/`full` preset or the `cpu` preset with 5 seeds.** Afterwards, changes go only in "Deviations" at the bottom.

## Revision history and DATA ALREADY SEEN (full disclosure)
* v1 (abstract random POMDPs) predicted "Mamba and Transformer beat RWKV". v2 moved to a grid-world hidden-object task and predicted "Transformer beats RWKV/Mamba on the patrol".
* **Seen before v3:** (a) a first CPU run by the author on the ABSTRACT envs (`delayed_readout`, `random_fast`, `random_slow`; 4 architectures x 3 seeds, 1500 steps, LR grid {1e-3,3e-3,1e-2}); (b) the LR-search files of that run; (c) a single-seed *grid* pilot (1500 steps, shared LR)
  in a sandbox: partially inspected (excess KL and probe values for some cells), never analysed statistically.
* What that data showed and how it changed the design (nothing below was tuned to make a hypothesis come out):
  1. 7 of 12 cells chose the TOP learning rate; the transformer's held-out CE was still falling at the edge => the architecture ranking was confounded by under-tuning. **Rule change:** adaptive, edge-extending LR search (`tuning.py`).
  2. `random_fast` is almost unpredictable (excess KL ~1e-3 vs entropy ~1.6) and the LR differences there were below the resolution of the validation set => **flat-grid rule** and a **skill score**.
  3. The state-belief probe ranked architectures differently from the predictive metrics (LSTM first on M2b, last-but-one on M1). Explanation: components of the belief that no prediction needs are not identifiable => **primary probe is now the predictive-state (PSR) probe**; the raw belief probe is secondary.
  4. With 3 seeds the permutation test cannot reach alpha=0.05 (min p = 0.1 after Holm) => **5 seeds minimum**; tests two-sided; the analysis states the minimum attainable p.
  5. On `delayed_readout` the order was Mamba < RWKV < LSTM << Transformer (excess KL); this contradicts the v2 direction (Transformer best) and is not evidence about the grid patrol (different memory structure).
* Anything computed on a grid preset with >=5 seeds after this commit is confirmatory; everything before is exploratory.

## Question
Do sequence models store the search history as a faithful belief over the hidden object's location, and does the *kind of memory* (in-place real decay vs dense recurrence vs lag-addressable) determine how faithful it is when the Bayes belief requires
rotating/shifting stored evidence? (THEORY.md §2, §7)

## Design (fixed)
* Grid (`grid.py`, 7x7 pillar map): `grid_patrol` (PRIMARY; P=16 ring, p_move=0.85), `grid_drift` (CONTROL; lazy walk, stay=0.5), `grid_static` (exploratory). Uniform-random agent. Exploratory robustness: `grid_patrol_seek`.
* Kalman pair (`lgssm.py`, matched noise and stationary signal variance, only the spectrum differs): `osc_rotating` (complex eigenvalues) vs `osc_decay` (real eigenvalues).
* Sanity (theory checks, not part of the question): `delayed_readout`, `random_fast`, `random_slow`.
* Architectures: `lstm` (dense recurrent), `rwkv`, `mamba` (real-diagonal recurrent), `transformer` (lag-addressable), `delay_mlp` (literal delay embedding, W=8: the classical null). Ablation: `mamba_noconv`.
* Budget: preset `full` (100k params, 10 000 steps, batch 64, L=64) for the grid; preset `full` of `run_lgssm` for the oscillators. 5 seeds (0-4).
* Learning rate: per (env, arch), held-out loss (oracle-free) at half length, start grid {1e-3,3e-3,1e-2,3e-2}, extended x3 (/3) while the best is on an edge (bounds 1e-4..1e-1), flat grid (spread < 2e-3) -> default. Runs whose LR is still on a bound, or whose excess KL fell >15% over the last 20% of training, are FLAGGED; primary tests are reported with and without flagged cells.
* Evaluation: fresh sequences, t >= burn-in; probes fit on 70% of sequences, tested on 30%; probe train scores stored to expose over-fitting.

## Hypotheses (all two-sided; exact permutation on the difference of means; Holm within each (env, metric) family of four contrasts; alpha=0.05)
Metrics: M1 excess KL (lower better), M2-psr predictive-state probe (higher better), M3 lag-profile W1 (lower better). Contrasts: lstm vs rwkv, lstm vs mamba, transformer vs rwkv, transformer vs mamba.
* **H1 (primary, `grid_patrol`).** The four contrasts differ from zero on M1, M2-psr, M3. *Expected direction from Prop 7/8: dense recurrent (LSTM) better than real-diagonal.* No direction is assumed for the transformer.
* **H2 (interaction; the theory test).** On `skill` and `probe_psr_score`, the gap [real-diagonal(rwkv,mamba) - lstm] on `grid_patrol` differs from the same gap on `grid_drift` (95% bootstrap CI over seeds excludes 0). *Expected: real-diagonal models fall further behind on the patrol => negative interaction on skill.*
* **H3 (Kalman pair).** Same interaction on `osc_rotating` vs `osc_decay` for `skill` and `kernel_rel_err` (lower better), with groups {rwkv, mamba} vs {lstm, delay_mlp}. *Expected: real-diagonal kernels miss the oscillation (large kernel error on osc_rotating only).*
* **H4 (exploratory).** `mamba` vs `mamba_noconv` (do the width-4 conv taps matter?); `delay_mlp` vs each learned model (does learned memory beat a literal window?); `grid_static`; `grid_patrol_seek`; belief-moment R^2; belief-map figures.
* **H5 (descriptive).** Capacity sweep {5k..80k params}, 3 seeds: smallest budget with mean excess KL < 0.01 per architecture. No test.

## What counts against the theory
1. No interaction in H2 and H3 (gaps are the same with and without rotation): the rotation/shift explanation is not supported.
2. `delay_mlp` matches or beats the learned recurrent models: learned memory brings nothing beyond a fixed window here.
3. M1, M2-psr, M3 disagree in sign on the ordering (the notion of "faithfulness" is not coherent).
4. Excess KL ~ 0 with low probe scores is a FINDING (prediction without a linearly decodable Bayes representation), not a failure.

## Known threats
* Parameter-matching does not match recurrent state size or FLOPs. One map/instance per family. Softmax-linear probes understate non-linearly encoded beliefs. Decodability is not use (no causal intervention on the hidden state yet).
* Prop 7/8 concern single linear real-diagonal layers; trained RWKV/Mamba have depth, gating, token-shift/conv taps.
* Transformer uses learned absolute positions with max_len=L; length generalisation is untested.

---
## Deviations (append only)
_none yet_
