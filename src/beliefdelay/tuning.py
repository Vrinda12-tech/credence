"""Learning-rate search that cannot silently clip at the edge of the grid.

Lesson from the first CPU run: 7 of 12 (env, arch) cells picked the TOP of {1e-3, 3e-3, 1e-2}, and the transformer's held-out CE was
still falling at the edge, so the architecture ranking was confounded by under-tuning.  Rules (fixed in PREREGISTRATION.md):

  1. evaluate the initial grid;
  2. if the best value is the largest (smallest) evaluated one, evaluate x`factor` (/`factor`) and repeat, up to [lo, hi];
  2b. never extend while the grid spread is below `tol`;
  3. if the whole grid is FLAT (max - min < tol nats: below what a finite validation set can resolve), use the value closest to `default`;
  4. record whether the result still sits on a bound.
"""
from __future__ import annotations

import math


def adaptive_lr_search(score_fn, grid, default=3e-3, tol=2e-3, lo=1e-4, hi=1e-1, factor=3.0, log=print):
    """score_fn(lr) -> held-out loss (lower is better).  Returns a JSON-serialisable dict."""
    scores = {}

    def ev(lr):
        lr = float(f"{lr:.6g}")
        if lr not in scores:
            scores[lr] = float(score_fn(lr))
            log(f"    tune lr={lr:g}: heldout loss {scores[lr]:.5f}")

    for lr in grid:
        ev(lr)
    extended = []
    while True:
        if max(scores.values()) - min(scores.values()) < tol:      # nothing resolvable: do not chase noise
            break
        best = min(scores, key=scores.get)
        if best == max(scores) and best * factor <= hi * 1.0001:
            ev(best * factor); extended.append(best * factor); continue
        if best == min(scores) and best / factor >= lo * 0.9999:
            ev(best / factor); extended.append(best / factor); continue
        break
    best = min(scores, key=scores.get)
    spread = max(scores.values()) - min(scores.values())
    flat = spread < tol
    if flat:
        best = min(scores, key=lambda x: abs(math.log(x / default)))
    at_bound = (not flat) and (best == max(scores) or best == min(scores))
    return dict(best=best, scores={str(k): v for k, v in scores.items()}, flat=flat, spread=spread,
                extended=extended, at_bound=at_bound)
