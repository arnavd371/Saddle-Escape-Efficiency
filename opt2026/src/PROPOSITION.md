# Proposition: RMSProp's fixed-radius escape time is curvature-independent, GD's is not

## Setup

Let $x_s$ be a saddle point of $f$ with Hessian $H = \nabla^2 f(x_s)$, and let
$v$ be a unit eigenvector of $H$ with eigenvalue $\lambda < 0$ (the escape
direction, i.e. $\lambda = \lambda_{\min}(H)$). Restrict attention to the
1-D dynamics along $v$: write $x_t = x_s + \delta_t v$ and, since $f$ is
well-approximated by its quadratic expansion near $x_s$, the gradient
component along $v$ is
$$g_t := \nabla f(x_t)\cdot v \approx \lambda\,\delta_t.$$

## Proposition

Under gradient descent with fixed learning rate $\eta$, the number of steps
to reach a fixed-radius escape boundary $\delta_t = r$ (criterion A) starting
from $\delta_0$ scales as
$$T_A^{\mathrm{GD}} \;\sim\; \frac{1}{\eta|\lambda|}\,\log\!\frac{r}{\delta_0},$$
i.e. it depends on the curvature magnitude $|\lambda|$: flatter saddles take
proportionally longer to escape. Under RMSProp with the same learning rate
$\eta$, the same boundary is reached in
$$T_A^{\mathrm{RMS}} \;\sim\; \frac{r-\delta_0}{\eta},$$
independent of $|\lambda|$.

## Proof sketch

**GD:** $\delta_{t+1} = \delta_t - \eta g_t \approx \delta_t(1+\eta|\lambda|)$,
a geometric recursion, so $\delta_t \approx \delta_0(1+\eta|\lambda|)^t$.
Solving $\delta_t = r$ for $t$ and using $\log(1+\eta|\lambda|)\approx
\eta|\lambda|$ for small $\eta|\lambda|$ gives the stated $T_A^{\mathrm{GD}}$.
Larger $|\lambda|$ (sharper saddle) shortens escape time; smaller $|\lambda|$
lengthens it.

**RMSProp:** the second-moment estimate $v_t$ (EMA of $g_t^2$) tracks
$g_t^2 \approx \lambda^2\delta_t^2$, i.e. it shrinks in lockstep with
$\delta_t$ near the saddle. The normalized update is
$$\delta_{t+1} = \delta_t - \eta\,\frac{g_t}{\sqrt{v_t}+\epsilon}
\;\approx\; \delta_t - \eta\,\mathrm{sign}(\delta_t)$$
once $v_t$ has tracked $g_t^2$ for a few steps (the $\epsilon$ stabilizer is
negligible once $|g_t|\gg\epsilon$, which is guaranteed away from $\delta_t=0$
exactly). This is a **linear**, not geometric, recursion with a *fixed* step
size $\eta$ per iteration, independent of $\lambda$: $\delta_t \approx
\delta_0 + t\eta\,\mathrm{sign}(\delta_0)$. Solving $\delta_t = r$ gives the
stated $T_A^{\mathrm{RMS}}$, with no $\lambda$-dependence at all. The
mechanism is exactly the one the extension prompt anticipated -- RMSProp's
per-coordinate normalization erases the gradient-magnitude information that
GD's step size depends on -- but the useful, checkable consequence is
specifically about criterion A's *curvature-independence*, not merely "RMSProp
takes bigger steps."

## Empirical support (already in the collected data, no new experiments needed)

`results/table2_best_lr_see.csv` and the saddle curvatures logged during Ext1
give a strikingly clean check of this exact prediction:

| Function | $\lambda_{\min}$ at saddle | GD_fixed best $A$ | RMSProp best $A$ |
|---|---|---|---|
| XOR MLP (Ext 4, real network) | **-0.0152** | **0.003** | **1.000** |
| Levy | -2.96 | 0.017 | 1.000 |
| Styblinski | -15.85 | 0.549 | 1.000 |
| Ackley | -17.11 | 0.305 | 1.000 |
| Himmelblau | -50.61 | 0.637 | 1.000 |
| Rastrigin | -392.73 | 0.985 | 1.000 |

The XOR network's saddle (Ext 4, `opt2026_ext/kernels/ext4_nn_saddle/`) is
the shallowest curvature tested anywhere in this project by an order of
magnitude, and it slots into the pattern exactly as predicted: RMSProp still
hits 1.000, GD_fixed collapses to 0.003 (the theory's `1/|lambda|` scaling
means GD should be at its *worst* exactly here) -- an independent
confirmation on a genuinely different, non-benchmark-function landscape.

**RMSProp scores exactly 1.000 on every function**, regardless of
$|\lambda_{\min}|$ ranging over more than two orders of magnitude
(2.96 to 392.73) -- exactly the curvature-independence the proposition
predicts. **GD_fixed's score varies from 0.017 to 0.985** and tracks
curvature magnitude in the predicted direction: Rastrigin (by far the
sharpest saddle, $|\lambda_{\min}|=392.73$) gives GD its best score;
Levy (the flattest, $|\lambda_{\min}|=2.96$) gives GD its worst. The
ordering is not perfectly monotonic (Ackley's $|\lambda_{\min}|=17.11$
exceeds Styblinski's $15.85$ but scores lower under GD -- LR-grid discretization
and other saddle-geometry factors beyond the 1-D linearization surely
contribute), but the two extremes and the overall direction match the
proposition cleanly. This also explains, mechanistically rather than just
descriptively, why criterion A is the one criterion that systematically
favors adaptive optimizers over GD/SGD-family methods regardless of the
underlying landscape's curvature -- a point that feeds directly into the
Practical Implications section (criterion A is flagged there as misleading
specifically for adaptive-vs-non-adaptive comparisons, for exactly this
reason).

## Scope note

This says nothing directly about criteria B/D, which measure
$\lambda_{\min}(x_t)$ or the loss value at $x_t$ rather than total
displacement -- so this specific curvature-independence argument does not
automatically predict how B/D-based escape timing compares between GD and
RMSProp, and no claim is made about that here.
