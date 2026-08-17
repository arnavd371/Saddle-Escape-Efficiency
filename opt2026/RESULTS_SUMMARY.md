# RESULTS_SUMMARY: OPT 2026 extension of Saddle Escape Efficiency

This summarizes what changed versus the original 4-optimizer / 5-function /
2D baseline (`results_final/`, untouched), and whether each of the paper's
three original Findings still holds under the expanded scope. All numbers
below are pulled directly from the committed CSVs in `results/`; see
`opt2026_ext/SANITY_CHECKS.md` for the verification work done before writing
this document.

**Framing note for the paper itself:** the honest headline across all three
extensions is not "the original findings replicate cleanly at larger scale."
It's closer to: *within-family stability replicates, between-family
disagreement replicates and sharpens, and B-D mutual validation is
partially confirmed rather than confirmed outright* -- it holds for two of
three functions tested at higher dimension and breaks down in a specific,
explicable way for the third. That is a stronger, more citable workshop
claim than "B-D always agrees," and it is consistent with the paper's
existing candor about messy results (the Rastrigin C-criterion saturation
discussion already sets this tone). We'd recommend the abstract/intro be
updated to say the dimensionality hypothesis is **partially confirmed**, not
confirmed outright, and that the Styblinski counterexample be presented as a
positive contribution ("B-D agreement is not universal, and here is a case
where it breaks down and why") rather than a caveat buried in a footnote.

---

## Paper structure recommendation for the Sept 4 submission

Given the 3.5-week timeline, not everything documented in this file belongs
in the main paper with equal weight. Recommended structure:

**Main paper (target: 4-8 pages):**
1. Core finding (baseline, unchanged) + Finding 1/2 replication at n=7
   (short -- these hold, they don't need much space)
2. **Finding 3, expanded**: B-D partially confirmed, Styblinski
   counterexample presented as a genuine contribution, not a caveat
   (see framing note above)
3. **Mechanistic proposition** (`opt2026_ext/PROPOSITION.md`): why RMSProp
   wins under criterion A regardless of curvature -- this is what upgrades
   the paper from "interesting observation" to "we understand why"
4. **NN saddle experiment** (`opt2026_ext/kernels/ext4_nn_saddle/`, complete
   -- see "Extension 4" below): partial replication -- RMSProp's curvature-
   independent criterion-A dominance replicates cleanly (independent
   confirmation of the proposition), but the Ackley/Rastrigin-style A-B
   rank inversion does not; this network's saddle shows higher overall
   cross-criterion concordance (W=0.84) than any benchmark function tested
5. **Practical Implications** (`opt2026_ext/PRACTICAL_IMPLICATIONS.md`):
   half a page, decision flowchart, serves the workshop's practitioner
   audience directly
6. Lion LR-convention artifact (short paragraph -- it's a clean, self-
   contained result: the paper-convention scaled LR substantially explains
   Lion's poor apparent performance)

**Dimensionality sweep: compress to one figure + one paragraph in the main
text**, not a full section. The headline (B-D holds for Ackley/Rastrigin,
breaks for Styblinski, flat across all dimensions tested) is stated once;
`results/fig7_dimensionality_sweep.png` and Table V go in main text or an
appendix depending on final page budget, but don't re-derive the mechanism
at length -- it's already covered by the Finding-3 discussion above.

**Ext 3 (curvature sweep): move to an appendix**, framed explicitly as
inconclusive -- non-monotonic W-vs-k, correlation CI crosses zero,
underpowered per-W-estimate (not a bare null result, see SANITY_CHECKS.md
Check 2) -- "we also explored this, here's why it's inconclusive, future
work." Do not present it as a main contribution; a null result with wide
CIs presented as headline material weakens the paper's overall claims.

**Priority order if time runs short:** the NN experiment carries the most
execution uncertainty and was started first for that reason (see
SANITY_CHECKS.md-adjacent work in `opt2026_ext/verification/` for the real
debugging this required: global saddle-search fails on saturating
activations, required a constructive symmetry-based approach instead). If
it fails or is inconclusive, the paper is still submittable as: core finding
+ Lion artifact + Styblinski counterexample + mechanistic proposition +
practical implications, in 4 clean pages -- a solid poster. The NN result,
if it lands cleanly, is what pushes toward a stronger poster or an oral.

---

## What changed

| | Baseline | Extension |
|---|---|---|
| Optimizers | 4 (GD, Adam, RMSProp, AdaGrad) | 7 (+ AdamW, SGD+Nesterov, Lion) |
| Functions | 5, all 2D | Same 5 in 2D, + Ackley/Rastrigin/Styblinski at n=2,5,10,25,50 |
| Curvature story | 5-point anecdote (one point per function) | 18-point continuous sweep on a verified synthetic saddle family |
| Kendall's W | point estimate only | bootstrap 95% CI on every W |
| Compute | local/CPU (implied) | Kaggle T4 GPU kernels, checkpointed per dimension |

All new work is in `opt2026_ext/`; `results_final/` and the original
notebook are untouched, so the original tables remain independently
reproducible.

---

## Finding 1: within-family threshold stability -- **holds**

Mean Spearman rho between extreme parameter values, same family, n=7
optimizers (`results/within_family_stability_ext.csv`):

| Family | mean rho |
|---|---|
| A (radius) | 0.93 |
| B (curvature) | **1.00** |
| C (eigen-disp) | 0.64 |
| D (loss-drop) | 0.87 |

Consistent with the baseline's n=4 result: criteria are internally stable
across their own parameter sweep regardless of how many optimizers are
compared. Family C is the least stable of the four, same as at baseline.

## Finding 2: between-family disagreement -- **holds, and is now statistically legible**

Kendall's W across all 4 criteria, n=7 optimizers, with bootstrap 95% CI
(`results/table3_kendall_w.csv`):

| Function | W | 95% CI |
|---|---|---|
| Himmelblau | 0.62 | [0.14, 0.84] |
| Ackley | 0.48 | [0.14, 0.76] |
| Rastrigin | 0.45 | [0.16, 0.73] |
| Styblinski | 0.60 | [0.05, 0.86] |
| Levy | 0.50 | [0.16, 0.71] |

Range 0.45-0.62, versus 0.11-0.47 at baseline (n=4, no CI). The W values
themselves shifted up somewhat with more optimizers in the pool, but the
qualitative story is the same: moderate, not high, concordance across
criteria on every function. The CIs are wide (n_items=7 is still a small
number of "raters" for a concordance coefficient) and all of them comfortably
exclude both W=0 and W=1 except Styblinski's, whose lower bound (0.047) sits
right at the edge -- read that one function's W with extra caution.

## Finding 3: B-D mutual validation -- **partially confirmed, not universal**

This is the headline change. At baseline (n=4, 2D only): B-D rho = +1.00 on
every one of 5 functions, reported as "geometry-based mutual validation."

At n=7 (still 2D, `results/table4_pairwise_spearman.csv`):

| Function | B-D rho |
|---|---|
| Himmelblau | +0.89 |
| Ackley | +0.96 |
| Rastrigin | **+1.00** |
| Styblinski | **+0.46** |
| Levy | +0.94 |

Adding three more optimizers already breaks "always +1.00" -- Himmelblau
drops to +0.89 and **Styblinski drops to +0.46**, a real disagreement, not
noise (see verification below).

## Extension 2: does the dimensionality hypothesis hold?

The baseline paper hypothesizes geometry-grounded criteria (B, D) should keep
agreeing regardless of dimension, while distance-based criteria (A) should
drift further from them as dimension grows. Tested on Ackley, Rastrigin,
Styblinski-Tang at n=2,5,10,25,50 (`results/table5_dimensionality_summary.csv`,
`results/fig7_dimensionality_sweep.png`):

- **B-D holds the hypothesis exactly for Ackley and Rastrigin**: rho stays
  at 0.96-1.00 across every dimension tested, 2 through 50.
- **B-D does NOT hold it for Styblinski**: rho sits at 0.43-0.46 at every
  dimension, 2 through 50 -- flat, not degrading, but never near 1.0 either.
- **A-B shows no universal pattern.** Ackley's |A-B rho| climbs cleanly from
  0.07 (n=2) to 0.87 (n=50) -- consistent with "distance drifts from
  curvature." Rastrigin's is non-monotonic (0.25 -> 0.50 -> 0.21). Styblinski's
  declines mildly (0.89 -> 0.61). The "distance-based criteria drift further"
  half of the hypothesis is a real pattern for one function, not a general law.

**Verdict: the hypothesis is partially confirmed.** Geometry-grounded mutual
validation (B-D) is dimension-independent where it holds at all -- but
whether it holds is a property of the function's saddle geometry, decided
already at n=2, not something that emerges or degrades with dimension. See
`opt2026_ext/SANITY_CHECKS.md` Check 3 for why this is a distinct claim from
the Ext 3 curvature-sweep result below, not the same phenomenon in disguise.

### Why does Styblinski break B-D specifically?

Investigated during verification (Check 1 in SANITY_CHECKS.md): at n=50,
GD_fixed and SGD_Nesterov escape 200/200 trials under criterion B but 0/200
under criterion D. They satisfy the curvature-exit condition
(`lambda_min > -eps`) quickly, but never clear the additional loss-drop
threshold (`f < f_s - 0.5`) within T_max=200 steps. That is a real
dynamical effect, not a numerical artifact: those two optimizers curve back
toward flat local curvature (satisfying B) faster than they make absolute
loss progress on Styblinski-Tang's quartic landscape (never satisfying D).
This mechanism is present already at n=2 and does not depend on the fix
described below.

### A numerical caveat on this specific number

A real bug was found and fixed while producing this result: the batched
Lanczos Hessian-eigenvalue estimator (needed for n>2, since the closed-form
2x2 formula doesn't generalize) crashed on a diverged Styblinski trajectory
at n=2. The fix sanitizes the Lanczos recurrence against overflow. Local
verification (CPU, see `opt2026_ext/verification/`) found:
- The qualitative finding (B-D rho far from 1.0 for Styblinski, at every
  dimension) is robust to the fix's specific clamp choice, and in fact
  **predates the fix entirely** -- Ext 1's 2D result (B-D=+0.46 at n=2) comes
  from the original closed-form `beigs()` code path, never touched by the
  Lanczos fix.
- The precise decimal at n=50 has real sensitivity to clamp choice: 0.429
  (as reported, deployed clamp) vs 0.536 (permissive clamp, local CPU
  replay) -- both clearly far from 1.0, neither near it. Report this number
  as **B-D rho approximately 0.43-0.54 at n=50**, not a single fixed decimal,
  in the same "read with appropriate caution" spirit the baseline paper
  already uses for small-n coefficients.

## Extension 3: continuous curvature-sharpness sweep

Replaces the informal 5-point Rastrigin-only anecdote with an 18-point sweep
over a parametrized synthetic saddle family, `a*x^2 - b*y^2 + k*sin(wx)*sin(wy)`,
verified analytically and numerically to remain a genuine saddle at the
origin for every k tested (`results/curvature_saddle_verification.csv`: 18/18
accepted, gradient exactly zero, finite-difference cross-check agrees to
2.6e-5). Result (`results/fig8_curvature_sweep.png`,
`results/table6_curvature_summary.csv`):

- Kendall's W vs. k is **non-monotonic**: rises from 0.28 (k=0, perfectly
  smooth) to a peak of ~0.78 around k=4, then falls back to ~0.45 by k=10.
  Not the clean "W decreases as curvature gets messier" story a reader might
  expect.
- Spearman rho(k, W) = -0.36, n=18 k-values, 95% bootstrap CI [-0.82, +0.26].
  The CI crosses zero -- **report this as underpowered, not as a null
  result.** The underpowering traces specifically to each individual W(k)
  estimate being noisy (only n_items=7 optimizers feed each W; per-k 95% CIs
  are themselves 0.35-0.5 units wide, comparable to the whole W range
  observed across the sweep), not to having too few k-values -- 18 conditions
  is already well beyond the baseline's n=5 anecdote. See SANITY_CHECKS.md
  Check 2 for the full breakdown.

## Extension 4: does the pattern replicate on a real neural network?

The highest-priority, highest-uncertainty addition: does the criterion-
dependence phenomenon show up outside synthetic benchmark functions, on a
real (if tiny) network's parameter-space loss landscape? Setup: a 2-2-1
tanh/sigmoid MLP on XOR (9 parameters), BCE loss, saddle found via a
constructive tied-unit-symmetry method (global multi-start `fsolve` -- the
method that works for the benchmark functions -- fails here; see
`opt2026_ext/xor_network.py`'s docstring for the real debugging this took).
Full results in `results/ext4_xor_*.csv`.

**What replicates cleanly:** RMSProp again scores **exactly SEE_A = 1.000**
while GD_fixed is a near-total failure (0.003, 0.0004, 0.000, 0.0001 across
A/B/C/D) -- an independent confirmation of the mechanistic proposition
above, on a genuinely different (real, non-benchmark) landscape. This
saddle happens to be the shallowest curvature tested anywhere in this
project (lambda_min=-0.0152, versus -2.96 for the shallowest benchmark
saddle, Levy), so GD's near-total failure is exactly what the proposition's
1/|lambda| scaling predicts.

**What does NOT replicate:** the Ackley/Rastrigin-style A-B rank
*inversion* (negative Spearman correlation) does not appear here. A-B rho
= **+0.96**, and in fact all 6 pairwise criterion correlations are
positive (0.67-0.96, `results/ext4_xor_spearman.csv`). Kendall's W = **0.84**
[0.38, 0.91] -- higher concordance across all 4 criteria than on any of the
5 benchmark functions (0.45-0.62 range).

**Honest reading:** criterion-dependence itself (different criteria giving
very different absolute SEE values, e.g. RMSProp's 1.000 under A vs. 0.015
under D) is still clearly present. But the specific *rank-inversion* flavor
of disagreement that motivated some of the baseline paper's most dramatic
claims is landscape-specific, not universal -- on this one real network's
saddle, criteria broadly agree on *ranking* even while disagreeing sharply
on *magnitude*. This is preliminary evidence from a single network/single
saddle, not a general claim about neural networks -- but it is a genuine,
citable contribution either way, exactly per the original framing: the
result didn't need to be dramatic to be useful, and this one is
informative precisely because it's a partial, not total, replication.

## Do the new optimizers behave as expected?

`results/adamw_lion_pattern_check_ext.csv`, `results/lion_lr_robustness_check.csv`:

- **AdamW tracks Adam almost exactly**: mean |SEE difference| <= 0.006 across
  all 4 criteria, on every function. The decoupled weight decay (wd=1e-2,
  PyTorch's own default, not hand-tuned -- see the docstring in
  `opt2026_ext/optimizers.py` for why this choice has no principled
  loss-landscape meaning on toy 2D problems) barely moves these trajectories.
- **Lion does NOT pattern with RMSProp**, contrary to the sign-based-update
  hypothesis in the original prompt: mean |SEE difference| vs RMSProp ranges
  0.28-0.99 across criteria -- the opposite of tracking closely. However,
  this largely reflects the paper-convention LR scaling (Lion swept at
  1/10th the shared LR grid, per the Lion paper's own recommendation): at
  the *unscaled* LR grid (same LRs as every other optimizer), Lion's mean
  best-LR SEE roughly triples, from 0.156 to 0.429
  (`lion_lr_robustness_check.csv`). Lion's poor showing under the paper
  convention is substantially an LR-tuning artifact, not an intrinsic
  property of sign-based updates -- report both numbers, not just the
  paper-convention one.

## Limitations carried over / new

- All the baseline's existing "Known limitations" still apply (GPU
  non-determinism vs CPU, C's saturation on some functions, n=4-analogue
  small-sample caution where relevant).
- New: the nD Lanczos Hessian-eigenvalue estimator (needed for n>2) is
  validated to 1e-13 accuracy against a dense reference at small n
  (`opt2026_ext/kernels/ext2_diagnostic/`), but its numerical-safety
  clamp has a measurable (not huge) effect on exact SEE values for
  optimizers whose trajectories reach extreme magnitudes -- flagged
  specifically for Styblinski n=50's B-D number above.
- New: Lion's reported SEE depends heavily on which LR convention is used;
  both the paper-convention (10x-scaled) and unscaled numbers are reported
  above rather than picking one.
- Ext 2 dimensionality sweep and Ext 3 curvature sweep both used N=200
  trials, T_max=200 steps, seed=42, matching the baseline's discipline
  throughout, with all raw per-trial escape/step data saved under
  `results/raw/` so every table here is regenerable without rerunning
  any simulation (`opt2026_ext/make_tables_figures.py`,
  `opt2026_ext/make_dim_curvature_figures.py`).
