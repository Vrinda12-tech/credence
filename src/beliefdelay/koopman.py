"""Koopman / DMD-with-control spectral diagnostics: what LINEAR dynamics does a trained model's hidden state implement?

Koopman view.  For x_{t+1} = F(x_t) the Koopman operator acts on observables g by (K g)(x) = g(F(x)); it is linear even when F is not.
For a Markov kernel T it is the backward operator (K g)(s) = E[g(s')|s] = T g, whose adjoint on beliefs (measures) b -> b T is the
Perron-Frobenius operator: OUR BELIEF PREDICT STEP.  Eigenfunctions of K are the coordinates in which the dynamics is diagonal.
  * patrol ring:  eigenfunctions = Fourier modes on the ring, eigenvalues (1-p) + p exp(2 pi i m / P)  (complex)
  * decaying/drifting system: real eigenvalues;   * linear-Gaussian oscillator: eigenvalues of A (complex pair r e^{+-i w}).
For a filter driven by observations, the belief mean obeys m_{t+1} = A(I-KC) m_t + A K y_t: its 'Koopman spectrum with input' is eig(A(I-KC)).

Delay embedding <-> Koopman (Arbabi & Mezic 2017, Hankel-DMD): the SVD of a Hankel matrix of delay coordinates spans a Koopman-invariant subspace,
so the delay embedding IS a finite-dimensional Koopman representation.  A diagonal-real recurrence has a real Koopman spectrum by construction;
a rotating system needs complex eigenvalues.  `fit_dmdc` measures which spectrum the hidden state actually uses (probe-free).

Caveat: hidden states of nonlinear networks are only approximately linear-driven; the fit's R^2 is reported and must be read alongside the eigenvalues.
For a transformer the residual stream at position t is not a recurrent state, so its 'spectrum' is descriptive only.
"""
from __future__ import annotations

import numpy as np
import torch


def fit_dmdc(H: torch.Tensor, U: torch.Tensor, rank: int = 4, burn: int = 16, ridge: float = 1e-8) -> dict:
    """Fit h_t = A h_{t-1} + B u_t on PCA-reduced hidden states.  H (N,L,d), U (N,L,m).  Returns eigenvalues of A and fit R^2.
    `rank` is clamped to min(rank, d, #timesteps-1, #singular values) so a low-dimensional or short-sequence hidden
    state (e.g. a small ablation model) degrades gracefully instead of producing a silently wrong reshape."""
    H, U = H.double(), U.double()
    N, L, d = H.shape
    if L - burn < 2:
        raise ValueError(f"fit_dmdc: burn={burn} leaves fewer than 2 usable timesteps out of L={L}")
    Hf = H[:, burn:].reshape(-1, d)
    mu = Hf.mean(0, keepdim=True)
    _, S, Vt = torch.linalg.svd(Hf - mu, full_matrices=False)
    rank = max(1, min(rank, d, len(S), L - burn - 1))
    V = Vt[:rank].T
    Z = (H - mu) @ V                                            # (N, L, r)
    X = torch.cat([Z[:, burn - 1:-1], U[:, burn:]], -1).reshape(-1, rank + U.shape[-1])
    Y = Z[:, burn:].reshape(-1, rank)
    W = torch.linalg.solve(X.T @ X + ridge * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y)
    A = W[:rank].T
    sse = ((Y - X @ W) ** 2).sum()
    sst = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    ev = np.linalg.eigvals(A.numpy())
    ev = ev[np.argsort(-np.abs(ev))]
    return dict(eig=ev, A=A.numpy(), r2=float(1 - sse / sst), explained_var=float((S[:rank] ** 2).sum() / (S ** 2).sum()))


def spectral_match(eig_fit: np.ndarray, eig_true: np.ndarray) -> dict:
    """For each true eigenvalue, distance to the nearest fitted one (complex plane); plus whether the fit contains any rotation."""
    d = [float(np.min(np.abs(eig_fit - e))) for e in eig_true]
    rot = bool(np.any((np.abs(eig_fit.imag) > 0.15) & (np.abs(eig_fit) > 0.5)))
    lead = eig_fit[np.argmax(np.abs(eig_fit.imag))]
    return dict(spec_dist=float(np.mean(d)), spec_dist_each=d, has_rotation=rot,
                lead_complex=complex(lead), lead_angle_deg=float(np.degrees(abs(np.angle(lead)))))
