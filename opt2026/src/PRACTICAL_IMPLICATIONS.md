# Practical Implications

Computing a full Hessian at every optimizer step is impractical at any
realistic scale. Given that constraint, what should a practitioner actually
use to decide "has this optimizer escaped the saddle"?

**Criterion A (fixed radius) is fast but systematically misleading when
comparing adaptive against non-adaptive optimizers.** The mechanistic
proposition above shows why: RMSProp's normalized step size is
curvature-independent (O(η) per step regardless of local curvature), while
GD's is curvature-dependent (O(η|λ|) per step). In the collected data,
RMSProp scores exactly 1.000 on best-LR SEE_A across all 5 functions
(λ_min ranging over two orders of magnitude), while GD_fixed ranges from
0.017 to 0.985 and tracks curvature magnitude. That is not RMSProp "escaping
better" in any sense a curvature-based criterion would recognize -- it is an
artifact of what fixed-radius distance measures. **Avoid criterion A, or any
purely distance-based criterion, when the comparison set includes both
adaptive and non-adaptive optimizers.** It is fine for comparing optimizers
within the same family (e.g. GD vs. GD+Nesterov), where the confound doesn't
apply.

**Criterion C (eigenvector projection) is cheap but not discriminative.**
49% of all (function, optimizer) combinations in the expanded n=7 dataset
score SEE_C = 1.000 exactly (17 of 35), and it has the lowest within-family
parameter stability of the four criteria (mean ρ=0.64 vs. 0.87-1.00 for
A/B/D). It saturates because the projection threshold (`c · r_curv`) is
usually crossed almost immediately once a trajectory moves at all -- it
mostly tells you a trajectory moved, not that it escaped a curvature basin.
**Don't use C as a primary criterion; it is at best a cheap sanity check
that a trajectory isn't frozen in place.**

**Criterion B (curvature exit) is the most principled criterion when you
can afford it**, but it requires an eigenvalue estimate of the Hessian at
every step -- one full Hessian-vector product minimum per step if using the
Lanczos estimator this project built for n>2 (`hessian_eig_nd.py`), more if
using finite differences. This is exactly what "escaped the saddle" should
mean: the local curvature actually flipped sign.

**Criterion D (loss-drop + cheap curvature check) is the most tractable
principled default.** It's B's curvature check plus a loss-drop
requirement, and B and D agree closely on 3 of 5 benchmark functions
(ρ≈0.89-1.00) -- so D inherits most of B's principled character while being
easier to reason about operationally (a practitioner already logs the loss
every step; adding one Lanczos HVP-based λ_min check per step is a modest
addition, not a new instrumentation burden). The important caveat, directly
from this project's own results: **D is not universally reliable either** --
it disagreed sharply with B on Styblinski-Tang (ρ≈0.43-0.46, both at
baseline and across every dimension tested), because its fixed absolute
loss-drop threshold (Δ=0.5 in this project's convention) doesn't scale with
a landscape's natural loss magnitude. **Recommendation: set Δ relative to
the loss value at the saddle itself (e.g. Δ = 0.1·|f(x_s)|), not as a fixed
absolute constant, when moving to a new landscape** -- this project used a
fixed Δ across very differently-scaled functions specifically to keep
criteria comparable within itself, but a practitioner tuning D for a single
real problem should not repeat that choice uncritically.

## Decision flowchart

```
Can you afford >=1 Hessian-vector product per optimizer step?
├── Yes  -> use criterion B (curvature exit) directly.
│           If the comparison includes both adaptive and non-adaptive
│           optimizers, do NOT also report A as if it were comparable.
└── No   -> use criterion D (loss-drop + a single cheap Lanczos lambda_min
            check), with Delta scaled to the saddle's own loss magnitude,
            not a fixed absolute constant.
            Cross-check against B on a small pilot run if possible --
            this project's Styblinski result shows B-D agreement is a
            property of the specific landscape, not guaranteed.

Never use A alone to compare adaptive vs. non-adaptive optimizers.
Never rely on C as more than a "did anything happen" sanity check.
```
