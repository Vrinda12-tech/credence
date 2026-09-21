# THEORY.md

Status labels: **[Proved]** short proof or standard fact; **[Tested]** checked numerically in `tests/`;
**[Motivation]** a heuristic that suggests the hypothesis but is *not* a theorem about trained networks.

## Research question

An agent explores a map and must keep track of a **hidden object** using only a noisy proximity sensor. The Bayes-optimal
memory of that search is a **belief map** `b_t(c) = P(object at cell c | everything seen so far)`.

> **How do sequence models store the history of a hidden-object search — as a belief map that is updated, shifted and
> decayed the way Bayes prescribes, or as some other code — and does memory that is addressable by *lag* (delay-embedding-like)
> give a more faithful map than memory that only decays in place?**

Next-observation prediction is only the *training signal*. The object of study is the internal belief map, compared cell by
cell with the exact one. (Belief maps can be drawn: `python -m beliefdelay.viz`.)

## 0. Setting (`grid.py`)

Map with free cells and walls; agent position `p_t` is observed; 5 actions (up/down/left/right/stay, walls block) chosen
by a history-dependent policy (uniform random, or an ε-greedy belief-following "seeker"). Hidden object cell `s_t` moves
independently of the agent with kernel `T`. Sensor `ℓ(r | p, s)` gives a reading `r ∈ {silent, near, here}` from the Manhattan
distance `|p − s|` (every entry > 0). Exact filter:

    b_t ∝ [ (b_{t-1} T) ⊙ ℓ(r_t | p_t, ·) ],      p*_t(r) = Σ_c (b_t T)(c) ℓ(r | p_{t+1}, c),   p_{t+1} = move(p_t, a_t).

**Prop 1 [Proved, standard].** `b_t` is a sufficient statistic of the history; `p*_t` is linear in `b_t`.
Because the object does not react to the agent, the filter is exact for *any* policy that depends only on the history
(`test_seeker_policy_is_valid…`). The filter agrees with brute-force path enumeration to 1e-10 (`test_filter_matches_brute_force_all_kinds`).

**Prop 2 [Proved].** For any predictor `q`, `E[−log q(r_{t+1})] = H(p*_t) + KL(p*_t ‖ q)`. Training on plain cross-entropy therefore
minimises `E KL(p*‖p_θ)` without the network ever seeing the belief.

**Identifiability [Proved, trivial] and its classical fix.** Beliefs that induce the same law over all futures are indistinguishable by any prediction loss, so probing the raw belief map penalises a model that compresses correctly.
The classical remedy is the *predictive state representation* (Littman–Sutton–Singh 2001): probe the vector of multi-step predictions the belief implies under each action (`env.test_vector`; k ≤ 3, one block per action×horizon). That is the primary probe (M2-psr); the raw map probe is secondary. An oracle whose state *is* the belief scores ≈0.96 on the map probe (0.955 measured), chance ≈ 0.

## 1. Three object behaviours = three memory demands

**Static object (`grid_static`) [Proved].** With `T = I`,
`log b_t(c) = log μ0(c) + Σ_{τ≤t} log ℓ(r_τ | p_τ, c) + const`. The belief map is an **additive accumulator** of log-evidence; every
reading counts equally regardless of its lag. No finite window is exact; the window-filter error `ε(W)` decays slowly
(measured: 0.15 → 0.0098 for W = 1 → 24).

**Drifting object (`grid_drift`) [Motivation + measured].** `T` is a lazy random walk (reversible, second eigenvalue `λ₂ < 1`), so old
evidence is forgotten geometrically: an *exponential* memory kernel is the right inductive bias. This is the control in which a fixed
decay should be adequate (measured `ε(1)=0.053 → ε(24)=0.002`).

**Patrolling object (`grid_patrol`) [Proved for p_move = 1; Tested].** The object advances along a cyclic route of `P` cells (here P = 16, the ring
around a pillar) w.p. `p_move` per step (0.85 in the experiments). For `p_move = 1` write `e_τ(i) = ℓ(r_τ | p_τ, route[i])`. The object's route index at time `τ` is
`i_τ = φ + τ` for an unknown constant phase `φ`, so a change of variables gives

    log b_t(route[j]) = Σ_{k=0}^{t} log e_{t−k}( (j − k) mod P ) + const.        (Prop 6)

**The belief map is a sum over lags `k` of the evidence vector seen `k` steps ago, cyclically shifted by `k`** — delay coordinates carrying a
lag-dependent shift (`test_patrol_belief_is_delay_shifted_evidence`, exact to 1e-10). For `p_move < 1` the phase diffuses: evidence both
shifts (by ≈ `p_move·k`) and loses precision — Prop 6 plus forgetting, which the exact filter handles.

## 2. Why architectures might differ on the patrol (motivation only)

Write the log-belief in Prop 6 as a state `x_t ∈ ℝ^P`: `x_t = S x_{t−1} + u_t`, with `S` the cyclic shift and `u_t` the new evidence.

**Prop 7 (rotations need complex eigenvalues) [Proved; narrow scope].** `S` is diagonalised by the DFT with eigenvalues the P-th roots of
unity, non-real for `P > 2`. A single linear recurrence with a **real diagonal** transition has only real eigenvalues, so its impulse responses are sums of
real exponentials and it cannot carry a shift/rotation as its own dynamics. Complex-diagonal SSMs (S4/LRU-style) and **dense** recurrences (LSTM) can.
The `mambapy` implementation and RWKV-4 use real per-channel decays.
*Scope:* depth, gating, nonlinearities, positional inputs and the read-out can circumvent this; it predicts nothing about a trained network by
itself. It is the sharpest reason to *expect* the patrol to separate architecture classes.

**Prop 5 (cost of a pure delay in a fixed-decay memory) [Proved; narrow].** For a linear time-invariant diagonal recurrence with `d` channels the Hankel matrix
`[g(i+j)]` of its impulse response has rank ≤ `d`; a pure delay `g(k)=δ_{k,τ}` has Hankel rank `τ+1`, so it needs `d ≥ τ+1` channels.

**Architecture classes (defined by mechanism, not by outcome):**

| class | members | how memory is addressed |
|---|---|---|
| **L** lag-addressable | Transformer (GPT-2) | attention over positions can index "the reading `k` steps ago" directly; a full context *is* a delay embedding of length `T` (cost `O(T)` state) |
| **R** dense recurrent | LSTM | full recurrent matrix (can realise permutations/rotations), input-dependent gates |
| **D** real-diagonal recurrent | RWKV-4, Mamba (mambapy) | per-channel real decay. RWKV's WKV is a softmax with a *fixed linear positional bias* `−(t−1−i)·w_c`; Mamba's kernel is `exp(A·Σ Δ_r)`, an exponential in learned intrinsic time. Both have short built-in delay taps (token-shift / width-4 conv); `mamba_noconv` ablates them |

**Hypothesis H (v3, see §7).** The gap between real-diagonal memory (class D) and dense-recurrent memory (class R) depends on the *spectrum* of the hidden dynamics: it appears when the Bayes belief must rotate/shift stored
evidence (patrol; oscillator) and not when exponential forgetting is exact (drift; decaying system). No direction is assumed for the lag-addressable class L.

## 3. Faithfulness metrics (all against the exact filter; `metrics.py`)

| id | quantity | ideal |
|----|----------|-------|
| M1  | `E KL(p*‖p_θ)`, excess predictive KL | 0 |
| M2a | probe score for the next-reading law | 1 |
| M2b | probe score for the **belief map** `b_t` (softmax-linear probe on the last hidden layer, held-out sequences) | 1 |
| M2c | `map_mass_at_mode` (Bayes mass on the decoded map's most-likely cell ÷ Bayes' own max) and `map_entropy_corr` (does the network know how uncertain it is?) | 1, 1 |
| M3  | `W1(Ĩ_θ, Ĩ*)`: lag profile `I(k)=E TV(p(·|h), p(·|h with the reading at t−k replaced))`, normalised; model vs exact | 0 |
| M4  | `L_eff`: window length `W` whose optimal window-filter error `ε(W)` equals the model's excess KL | larger = closer to long-memory Bayes |

## 4. Abstract sanity environments (kept for theory checks only)

`delayed_readout` (shift register of iid bits; the action selects a readout lag) has an **exact** finite delay embedding.
**Prop 3 [Proved + Tested].** The Bayes predictor depends only on the last `max lag + 1 = 6` pairs (window error 1e-18 at `W=6`, > 0 at `W=5`;
proof: likelihood and posterior factorise over bits). `random_fast/slow` are Dirichlet-random POMDPs with a checked Dobrushin bound
`δ(T_a) ≤ 1 − mix`. They validate the machinery and run with `--envs delayed_readout`, but they are **not** part of the research question.

## 5. What this project does not claim

* That any trained network *is* a delay embedding; only that a measurable map/lag structure can be compared with the exactly known optimum.
* Anything about NetHack or other real environments; that is a follow-up conditional on a positive result here.
* That parameter-matched means state-size- or compute-matched (`d_model`, `n_params` are stored per run).

## 6. References (from memory; verify volume/pages before citing)

Åström 1965; Takens 1981; Sauer–Yorke–Casdagli 1991; Hochreiter & Schmidhuber 1997; Jaeger 2000 (OOMs); Littman–Sutton–Singh 2001 (PSRs);
Vaswani et al. 2017; Gu, Goel, Ré 2022 (S4; complex-diagonal SSMs); Gu & Dao 2023 (Mamba); Peng et al. 2023 (RWKV); Orvieto et al. 2023 (LRU);
Arbabi & Mezić 2017 (Hankel DMD); Shai et al. 2024 (belief-state geometry in transformer residual streams; closest prior probing work);
Kara & Yüksel (finite-memory POMDP approximation); Del Moral / van Handel (filter stability).


## 7. Classical grounding (added after the first CPU run)

**Hidden-state problems solved by classical mathematics** (analogies; motivation, not evidence):

| system | hidden state | observations | classical solution | counterpart here |
|---|---|---|---|---|
| Ceres 1801 (Gauss), Apollo navigation | orbital elements | sparse noisy angles | least-squares orbit determination -> Kalman filter | `grid_patrol`, `osc_rotating` |
| Bayesian search (Koopman 1956; wreck searches) | wreck location | detection sensor | posterior probability map, drift models | the whole grid task; `grid_static`, `grid_drift` |
| Lorenz-63 (truncated convection) | 3-D state | one coordinate | Takens delay embedding (deterministic, noise-free) | the delay-embedding hypothesis; `delay_mlp` |
| Cylinder wake, vortex shedding | velocity field | few probes | Kalman/EnKF; Hankel-DMD/ERA (Hankel rank = Prop 5) | `osc_rotating` (complex pair = shedding frequency) |
| Weather, ocean | atmospheric state | sparse | 4D-Var, ensemble Kalman filter | belief map over cells |

**Prop 8 (the Bayes filter is a selective linear recurrence) [Proved].** The unnormalised belief satisfies `rho_t = diag(l_t) T^T rho_{t-1}`, with `l_t = l(r_t | p_t, .)`: an input-dependent DIAGONAL gain composed with a fixed DENSE transport `T^T`
(this is the discrete Zakai equation; normalisation only rescales `rho`). A selective diagonal SSM (Mamba) can express the gain; the transport is what it lacks.
*Static object:* `T = I`, so the filter is exactly a selective diagonal recurrence (log-domain: an additive accumulator).

**Prop 9 (gain and transport do not commute) [Proved].** For the ring, `S D S^{-1}` = `D` with its diagonal cyclically shifted, which equals `D` iff `D` is constant. So a generic evidence gain is not diagonal in the basis in which the shift is (the DFT), and no single fixed basis diagonalises both
factors of the recurrence. This is the structural reason the dense transport is the hard part for diagonal recurrences (Lie–Trotter splitting picture).

**Prop 10 (Kalman memory kernels) [Proved for linear-Gaussian; Tested].** For the linear-Gaussian oscillator the optimal predictor is a linear filter, `mu*_t = sum_k g(k) y_{t-k}`, with `g` a sum of exponentials whose rates are the eigenvalues of the closed-loop matrix
`A(I-KC)`. `osc_rotating` (`A = r Rot(w)`) has a damped-oscillation kernel (4 sign changes in 16 lags); `osc_decay` (real spectrum) has a positive monotone kernel (`test_kernel_oscillates_iff_spectrum_is_complex`).
Exact identity: excess MSE = `E(mu*_t - f_t)^2`. Kalman == joint-Gaussian conditioning to 1e-8 (`test_kalman_equals_gaussian_conditioning`).

**What is and is not implied.** Props 8-10 say *where* a diagonal-real recurrence has to work harder; depth, gating, token-shift and conv taps can compensate, and attention sidesteps the recurrence altogether. Hence the two-sided tests and the interaction design (H2, H3).

## 8. What the first CPU run (abstract envs) already suggests, and what it cannot
Excess KL on `delayed_readout` ordered Mamba < RWKV < LSTM << Transformer with the LR possibly clipped for three of four cells; the transformer result is therefore uninterpretable until re-tuned. Nothing here bears on H1-H3.
