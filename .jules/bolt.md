## 2024-10-14 - PyTorch einsum vs bmm overhead in sequential loops
**Learning:** `torch.einsum` introduces non-negligible setup/parsing overhead when invoked repeatedly inside time-step sequential loops (e.g., Bayes filter rollout over time horizon $L$). Replacing `torch.einsum("bc,brc->br", prior, Lk)` or `torch.einsum("bn,bnm->bm", prior, O)` with explicit batched matrix multiplication (`torch.bmm(Lk, prior.unsqueeze(-1)).squeeze(-1)` or `torch.bmm(prior.unsqueeze(1), O).squeeze(1)`) speeds up the Bayes filter step by ~20% with exact floating-point equivalence.
**Action:** Prefer `torch.bmm` or explicit tensor contraction operations over `torch.einsum` inside hot python loop iterations over sequence time steps.

## 2025-05-18 - Replacing nested loop bmm with single pre-contracted GEMM
**Learning:** Calling `torch.bmm` inside nested loops over actions $A$ and horizons $K$ for indexed positions incurs repeated tensor allocation and call overhead. Computing reading likelihoods across all map cells upfront via a single 2D matrix multiplication per horizon (`pk @ self.Lk_mat`) and then gathering positions `p` in the inner loop speeds up `GridPOMDP.test_vector` by ~2.5x with exact floating-point equivalence.
**Action:** Lift static observation matrix contractions outside nested loops into a single batch matrix multiplication before indexing.

## 2025-05-19 - Pre-contracting action transitions and emission matrices into a single PSR operator
**Learning:** In `POMDP.test_vector`, sequentially propagating belief states through $A \times K$ loops (`prior = prior @ T[a]`, `prior @ O[a]`) repeatedly applies matrix multiplications on high-dimensional batch/sequence tensors `(B, L, N)`. Pre-contracting the linear transition-emission operators $T_a^k O_a$ upfront into a single static $(N, A \cdot K \cdot M)$ matrix $M_{\text{psr}}$ allows computing all predictive state projections in a single `post @ M_psr` matrix multiplication. This reduces overhead by ~35x with exact floating-point equivalence.
**Action:** For multi-horizon predictive vector computations with static linear transition/emission operators, pre-contract operators into a single block matrix before multiplying by the batch sequence state tensor.

## 2025-05-20 - Reusing exact Bayes filter trajectories across window length sweeps
**Learning:** Computing effective delay depth metrics (`window_error_curve`) over a list of window sizes $W$ called `env.window_filter(obs, act, W)` once for each $W$, re-running `env.filter(obs, act)` $1 + |W|$ times on identical sequence tensors. Allowing `window_filter` to accept an optional pre-computed `full_filter=(post_f, pred_f)` tuple and passing it during window sweeps reduces redundant filter rollouts from $|Ws| + 1$ down to 1, yielding a ~1.5x-1.8x speedup with exact floating-point equivalence.
**Action:** When computing multi-window/delay-embedding properties over sequence histories, pass pre-computed exact filter outputs to `window_filter`.

## 2025-05-21 - Batching lag perturbation evaluations in sequence influence profile computations
**Learning:** In `influence_profiles`, evaluating interventional influence $I(k)$ over $k=0 \dots k_{max}$ sequential iterations ran $k_{max} + 1$ separate forward passes for `model` and $k_{max} + 1$ separate rollouts for `env.filter`. Batching all $k_{max} + 1$ perturbed observation sequences into a single tensor `((kmax + 1) * n_seq, L + 1)` allows computing model output predictions and exact Bayes filter rollouts in a single batched pass, eliminating Python loop and call overhead.
**Action:** Concatenate interventional/perturbed sequence histories across perturbation lags into a single batch tensor prior to model forward inference and filter evaluation.
