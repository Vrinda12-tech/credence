## 2024-10-14 - PyTorch einsum vs bmm overhead in sequential loops
**Learning:** `torch.einsum` introduces non-negligible setup/parsing overhead when invoked repeatedly inside time-step sequential loops (e.g., Bayes filter rollout over time horizon $L$). Replacing `torch.einsum("bc,brc->br", prior, Lk)` or `torch.einsum("bn,bnm->bm", prior, O)` with explicit batched matrix multiplication (`torch.bmm(Lk, prior.unsqueeze(-1)).squeeze(-1)` or `torch.bmm(prior.unsqueeze(1), O).squeeze(1)`) speeds up the Bayes filter step by ~20% with exact floating-point equivalence.
**Action:** Prefer `torch.bmm` or explicit tensor contraction operations over `torch.einsum` inside hot python loop iterations over sequence time steps.

## 2025-05-18 - Replacing nested loop bmm with single pre-contracted GEMM
**Learning:** Calling `torch.bmm` inside nested loops over actions $A$ and horizons $K$ for indexed positions incurs repeated tensor allocation and call overhead. Computing reading likelihoods across all map cells upfront via a single 2D matrix multiplication per horizon (`pk @ self.Lk_mat`) and then gathering positions `p` in the inner loop speeds up `GridPOMDP.test_vector` by ~2.5x with exact floating-point equivalence.
**Action:** Lift static observation matrix contractions outside nested loops into a single batch matrix multiplication before indexing.
