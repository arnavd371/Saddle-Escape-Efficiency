# Pre-synthesis sanity checks

Run before drafting RESULTS_SUMMARY.md, per explicit request, to catch problems
that are cheap to find now and expensive to discover after tables are written.
All checks below are local CPU verification of already-committed Kaggle
results; no additional Kaggle GPU time was used.

## Check 1: Is Styblinski's B-D counterexample a side effect of the Lanczos fix?

**Question:** the Lanczos eigendecomposition fix (see `hessian_eig_nd.py`,
commit "Fix: Lanczos eigendecomposition crash...") was needed specifically for
Styblinski, and Styblinski is also the function producing the most surprising
result (B-D Spearman rho ~0.43-0.46 across ALL dimensions, vs ~0.96-1.00 for
Ackley/Rastrigin). Before reporting this as a genuine finding, verify the fix
isn't itself producing it.

**Method:** reran the exact saved n=2 and n=50 Styblinski saddle geometries
locally (CPU) across the full optimizer x LR grid, twice -- once with the
deployed fix's clamp (1e8) and once with an effectively-permissive clamp
(1e15, i.e. clamp essentially never engages for realistic values) -- and
diffed trial-level escape labels directly.

**Findings:**
- At n=2 (where the original crash occurred): IDENTICAL escape labels under
  both clamp settings, for all 7 optimizers x 6 LRs. The clamp never engages
  for well-behaved n=2 trajectories in this CPU replay. (Caveat: the original
  crash happened on GPU; torch.manual_seed(42) does not guarantee bitwise-
  identical trajectories between CPU and CUDA even in float64 -- this is
  already a documented limitation in the baseline README -- so a CPU replay
  cannot force the exact diverging trajectory that crashed on Kaggle.)
- At n=50: the clamp DOES matter, but only for GD_fixed and SGD_Nesterov --
  the two optimizers whose trajectories reach extreme magnitudes under some
  LRs in the grid. It never changes WHETHER a trial escapes (always 200/200
  either way for the affected optimizers under criterion B), only the STEP at
  which escape is recorded. This shifted the full-grid best-LR B-D Spearman
  rho at n=50 from 0.429 (as reported) to 0.536 under the permissive clamp in
  this CPU replay.
- **A much bigger, non-numerical driver of the B-D gap was found in the
  process:** GD_fixed and SGD_Nesterov escape 200/200 under criterion B but
  0/200 under criterion D at n=50, REGARDLESS of clamp choice. They satisfy
  the curvature-exit condition (lambda_min > -eps) quickly but never clear
  the additional loss-drop threshold (f < f_s - 0.5) within T_max=200 steps.
  This is a real dynamical effect, not a numerical artifact.
- **Most important finding from this check:** Styblinski's weak B-D agreement
  is NOT a dimensionality effect. Ext 1 (2D, 7 optimizers, no dimensionality
  sweep involved) already shows Styblinski B-D rho = +0.46 at n=2 baseline --
  present before any dimensionality sweep, and BEFORE the Lanczos fix was
  even written (Ext 1 uses the 2D closed-form `beigs()`, entirely unrelated
  code path). This rules out the fix as the source of the qualitative
  pattern: the weak B-D agreement is an intrinsic property of Styblinski-
  Tang's saddle geometry interacting with criterion D's fixed Delta=0.5
  threshold, not an artifact of the nD Lanczos estimator or its fix.

**Verdict:** the QUALITATIVE finding (Styblinski's B-D agreement is
substantially weaker than Ackley/Rastrigin, ~0.43-0.54 rather than ~1.0) is
robust to the clamp choice and predates the fix entirely (visible already in
Ext 1's 2D result). The PRECISE decimal value at n=50 has real sensitivity
(~0.43-0.54 range) to how a handful of divergent GD_fixed/SGD_Nesterov
trajectories are numerically resolved. RESULTS_SUMMARY.md and any table
reporting this number should carry that caveat explicitly, in the same
"read with appropriate caution" spirit the baseline paper already uses for
small-n coefficients.

## Check 2: Is the Ext 3 k-vs-W null result underpowered?

From `results/curvature_k_vs_W_correlation.csv`:
Spearman rho(k, W) = -0.361, n = 18 k-values, 95% bootstrap CI = [-0.820,
+0.257], B = 2000 resamples.

**n=18 k-values is not itself underpowered** for a Spearman correlation --
substantially more than the original paper's n=5 Rastrigin anecdote. The
wide CI (half-width ~0.54, LARGER than the point estimate's magnitude of
0.36) instead traces to a different source: each individual W(k) estimate
carries wide uncertainty in its own right, because Kendall's W at each k is
computed over only n_items=7 optimizers (see `curvature_kendall_w_vs_k.csv`
-- individual 95% CIs on W range roughly 0.35-0.5 units wide, e.g. k=0:
W=0.28 [0.06,0.43], comparable in width to the entire observed W range across
the whole sweep, 0.28-0.78). That per-point noise propagates into the k-vs-W
relationship. So: this is a genuinely underpowered test of the k-vs-W
correlation, but the underpowering comes from having only 7 optimizers per
concordance estimate, not from having too few k-values. RESULTS_SUMMARY.md
should say this explicitly -- "underpowered because W itself is noisy at
n_items=7, not because the k-sweep (n=18) is too coarse" -- rather than
reporting a bare "not significant."

## Check 3: Is the Styblinski flat-line related to the Ext 3 non-monotonicity?

**No, and this needs one explicit sentence in the summary so readers don't
conflate them.** They differ on every axis:
- **Function:** Ext 2/Styblinski uses the real Styblinski-Tang benchmark.
  Ext 3 uses a synthetic saddle family (a*x^2 - b*y^2 + k*sin(wx)*sin(wy))
  that is NOT Styblinski-Tang, just tuned to reach comparable curvature
  magnitude at its k-max.
- **Quantity:** Ext 2/Styblinski tracks a single pairwise Spearman rho (B
  vs D). Ext 3 tracks Kendall's W across all four criteria simultaneously.
- **Independent variable:** Ext 2/Styblinski varies dimension (2D-50D) at
  fixed criterion pair. Ext 3 varies a curvature-sharpness parameter k at
  fixed dimension (always 2D).

Per Check 1, Styblinski's B-D gap is present already at n=2 and stays roughly
flat through n=50 -- it is a fixed property of that one function's saddle,
not something that grows or shrinks with either dimension or oscillation
density. Ext 3's non-monotonic W-vs-k curve is a separate, curvature-driven
phenomenon on an unrelated synthetic family. Both results are "criteria can
disagree in structured, function/landscape-specific ways" but are not the
same result and should not be cited as evidence for one another.
