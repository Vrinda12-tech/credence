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


## 9. Koopman view (added after the oscillator results)
The belief predict step `b -> bT` is the Perron-Frobenius operator; its adjoint `f -> Tf` is the Koopman (backward) operator. **Koopman eigenfunctions are the coordinates in which the hidden dynamics is diagonal**:
on the patrol ring they are the Fourier modes with eigenvalues `(1-p) + p e^{2 pi i m/P}` (complex); for the oscillator the eigenvalues of `A`. For a filter driven by observations the belief mean has spectrum `eig(A(I-KC))` (`LinearGaussian.closed_loop_eig`):
for `osc_rotating` this is `0.505 +- 0.502i` (modulus 0.71, angle 44.8 deg) versus the system's `0.686 +- 0.686i` (modulus 0.97, angle 45 deg): **observations preserve the rotation frequency but shorten the memory**.
Delay embedding is a finite-dimensional Koopman representation (Hankel-DMD; Arbabi & Mezic 2017), which is why `delay_mlp` is the natural null.
`koopman.fit_dmdc` fits `h_t = A h_{t-1} + B y_t` to a trained model's (PCA-reduced) hidden states; `run_lgssm --spectrum DIR` compares its eigenvalues with the exact filter spectrum (probe-free). DMDc recovers the exact spectrum from the Kalman state to 4e-6 (rotating) / 3e-4 (real) (`test_dmdc_recovers_closed_loop_spectrum...`).
**Scope correction to Prop 7:** a real-diagonal recurrence cannot carry an exact rotation as its own dynamics, but over a FINITE horizon a damped cosine is approximated by a signed sum of enough real exponentials (Prop 5 cost). With d ~ 40 channels and lags <= 16 the observed kernel errors (RWKV/Mamba ~4%) show this is achievable; the separation should therefore be sought at long horizons (`osc_rotating_slow`, r = 0.995, L >= 128) and in exact state tracking, not at short lags.


## 10. Statistical validation as its own artifact
`evaluation/rigorous_validation.py` and `tests/test_rigorous_validation.py` treat the evaluation pipeline itself as something to be
proved correct, not merely run: every guard (exact permutation validity, Holm FWER control under arbitrary dependence, Monte-Carlo
p-value positivity, bootstrap coverage collapse below n=5, TOST equivalence, pre-registration hash-locking) is checked either by
exact enumeration over the full permutation orbit or by simulation with a tolerance derived from the cited theorem, and each checker
is also run against a deliberately wrong "mutant" implementation that must fail. This is what "the proof is the maths, programmed
well" means in this project: the statistics are not just applied, their correctness is itself a tested claim.


## 11. Making "the RNN learned it" explicit (explicitness.py)

A black-box objection to RNN hidden states (unstructured h_t in R^D, "memory drift," no guarantees, can't inspect) is
answered here not by asserting the opposite but by splitting the claim into four INDEPENDENT, separately falsifiable
axes, each checked against an exact ground truth computed elsewhere in this project:

| axis | question | instrument | ground truth |
|---|---|---|---|
| sufficiency | does h_t carry enough information to predict optimally? | excess KL / skill | exact Bayes/Kalman filter |
| decodability | is that information LINEARLY exposed in h_t? | PSR / belief probes | exact predictive-state vector |
| dynamical form | does h_t evolve approximately linearly, with a spectrum resembling the process's own? | DMDc fit R^2 + spectral match | env.koopman_eig() / closed_loop_eig() |
| stability | does a small parameter change stay bounded across rollout length ("memory drift", Luo et al. 2024)? | amplification growth_ratio | analytic toy-RNN bound (Prop 1, arXiv:2405.15384) |

No axis implies any other: a model can be sufficient while being a decodability black box (observed on delayed_readout:
Mamba had the best excess KL and among the worst raw belief-probe scores), or linearly decodable while its own update
law is nonlinear. `audit()` composes the instruments already built for other parts of this project (metrics.py,
koopman.py, amplification.py) rather than adding new unverified machinery.

**Calibration, not just plumbing.** `tests/test_explicitness.py` requires the audit to be right about cases with a
known answer: a fresh model that ignores its input must score near the floor on sufficiency and decodability
(`BlindModel`); a model whose hidden state literally IS the belief must score near ceiling. Building this calibration
found two real bugs, not test artefacts: (1) `koopman.fit_dmdc` silently mis-sized its basis when the hidden
dimension was smaller than the requested rank, corrupting a reshape; (2) `metrics.fit_probe`'s standardisation used
an absolute `1e-6` floor on feature scale, which blows up any exactly-zero-variance feature (e.g. a structurally
unreachable cell in a raw belief vector) into a huge, spurious input.

**A genuine methodological finding, not a bug.** Calibrating the PSR probe against an oracle showed its score is
0.058 on a RAW-probability belief and 0.592 on a LOG-probability belief for the identical oracle (train and test
scores agree either way, so this is not overfitting). A softmax-linear probe is naturally suited to representations
where the target is affine in the input; Bayesian evidence composes additively in log-space (Prop 8), so a raw
probability vector is a poor match for a softmax read-out even when the information is, in a real sense, entirely
present. **Consequence: a low PSR/belief-probe score on a trained network's hidden state is evidence about how that
information is encoded (log-additively vs. otherwise), not proof the information is absent.** This is now stated
explicitly rather than left as an unstated assumption behind the probe ceiling.

## 12. Design principles enforced by this project's architecture (not aspirations -- each is code + a test)
1. Every mathematical claim ships with the code that would falsify it (a proof, a brute-force check, or a simulation
   with a theory-derived tolerance) in the SAME commit as the claim.
2. Every measuring instrument is itself calibrated against a case with a known answer (an oracle, a blind model, a
   toy system solvable in closed form) before being trusted on a real model.
3. A "the network learned X" claim is never accepted from convergence alone; it is decomposed into sufficiency,
   decodability, dynamical form and stability, each measured and each falsifiable independently.
4. A metric's ceiling is measured, not assumed; "near 1.0" is asserted only where an oracle achieves it, and the
   actual measured ceiling is documented and used as the comparison point instead.
5. Statistical decisions (p-values, corrections, confidence intervals) are never computed ad hoc; they are routed
   through one validated module whose guards are themselves proved and mutation-tested (evaluation/rigorous_validation.py).
6. Confounds found in real runs (an under-tuned learning rate, a core/head learning-rate mismatch) are fixed in the
   infrastructure and disclosed in PREREGISTRATION.md's revision history, not silently absorbed into "the architecture is worse."


## 13. Controllable hidden states (structured.py) -- intervention, not inference

Sections 1-11 measure whether an existing black-box architecture happens to encode Bayes-relevant structure after
training. This section builds two cells where the classical objects THEMSELVES are the parameters, so they can be
set, frozen, ablated and intervened on directly rather than inferred with a probe.

**KoopmanCell**: h_t is k explicit complex modes z_k = r_k e^{i theta_k} in POLAR form -- the model's Koopman
spectrum IS (r_k, theta_k), not something fitted to h_t afterward. `set_spectrum(eig, freeze=True)` pins it exactly
to `env.koopman_eig()` / `env.closed_loop_eig()`; `ablate_rotation()` mechanically zeroes every theta_k, a CAUSAL
test of Prop 7 (a real-diagonal recurrence cannot carry rotation), not a correlational one.

**Experiment (osc_rotating, 800 steps, 3 conditions, same architecture and budget):**

| spectrum | excess MSE | trainable params | learned/frozen spectrum |
|---|---|---|---|
| learned freely | 0.00476 | 197 | 0.144-0.306j, 0.314-0.482j (nowhere near truth) |
| initialised at truth, then trained | 0.00185 | 197 | 0.521+0.480j, 0.449-0.543j |
| FROZEN at truth | **0.00081** | 193 | 0.505+0.502j, 0.505-0.502j (exact) |
| true closed-loop spectrum | -- | -- | 0.505+0.502j, 0.505-0.502j |

Freezing the spectrum to ground truth beats free learning by 5.9x with FEWER trainable parameters, and at this
budget gradient descent alone does not find anything close to the true spectrum on its own. This is a causal claim
(controlling the spectrum changes the outcome) that no amount of post hoc probing of a black box could make.

**NeuralBayesCell**: h_t IS a log-probability vector over n_state cells; the recursion has the exact functional form
of the Bayes filter (Prop 8) with a LEARNED transition kernel and evidence map. `inject(t, true_belief)` overwrites
h_t at a chosen position; `set_transition(T, freeze=True)` pins the transition kernel to the true env.T.

**Experiment (grid_drift, 600 steps, inject the true belief at t=20, then let the model's own recursion continue):**

| | t=19 (pre) | t=20 (injected) | t=21 | t=25 | t=30 | t=40 |
|---|---|---|---|---|---|---|
| LEARNED T, no injection | 2.632 | 2.617 | 2.616 | 2.643 | 2.661 | 2.628 |
| LEARNED T, WITH injection | 2.632 | **0** | 1.902 | 2.622 | 2.659 | 2.628 |
| TRUE T (frozen), no injection | 1.177 | 1.194 | 1.204 | 1.244 | 1.313 | 1.277 |
| TRUE T (frozen), WITH injection | 1.177 | **0** | 0.173 | 0.47 | 0.727 | 0.904 |

(KL to the exact belief; t=19 matches exactly with and without injection, confirming injection is causal -- it never
touches the past.) With a merely-trained transition kernel, a correction injected at t=20 is destroyed within one
step (t=21 is already close to baseline, t=25 onward indistinguishable). With the transition kernel frozen to the
TRUE env.T, the correction is still worth 1.4x at t=40, twenty steps later. This is a causal test end-to-end loss
alone cannot make: it asks whether the recursion, GIVEN a known-correct starting point, behaves like the true
process going forward, which a lucky end-to-end fit could satisfy without the recursion itself being right.

Both cells match `models.SeqModel`'s core interface (arch names `"koopman"`, `"koopman_pure"`, `"neuralbayes"`) and
so are trainable/evaluable with every tool already built for the four baseline architectures.
