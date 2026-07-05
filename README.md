# Saddle Escape Efficiency (SEE)

Code accompanying the paper *"Saddle Escape Efficiency: Introducing a Novel
Metric and Revealing Its Criterion-Dependence in Non-Convex Landscapes."*

This repository contains the full experimental pipeline used to compute SEE
across four independent escape criteria, four optimizers, and five
non-convex benchmark functions. All numbers reported in the paper's tables
were verified against this code by re-running the full pipeline end-to-end.

## What this measures

**SEE** (Saddle Escape Efficiency) is defined as:

```
SEE = P_esc / mean_escape_step
```

where `P_esc` is the empirical probability of escaping a saddle region
within `T_max` iterations, and `mean_escape_step` is the mean step at which
escape occurred, conditioned on successful escape.

A high SEE means an optimizer escapes the saddle both **reliably** and
**quickly**.

The central finding of this project is that SEE is **not a single number**:
its value, and even the ranking of optimizers it implies, depends heavily on
*how "escape" is defined*. This repo evaluates four independent escape
criteria on identical trajectories to demonstrate that dependence directly.

## The four escape criteria

| Criterion | Definition | Parameter swept |
|---|---|---|
| **A — Fixed radius** | `‖x_t − x_s‖ > r` | `r ∈ {1.5, 2.0, 3.0}` |
| **B — Curvature exit** | `λ_min(H(x_t)) > −ε` | `ε ∈ {1e-2, 1e-3, 1e-4}` |
| **C — Eigenvector projection** | `\|⟨x_t − x_s, v⟩\| > c · r_curv` | `c ∈ {0.5, 1.0, 2.0}` |
| **D — Loss drop** | `f(x_t) < f(x_s) − δ` and `λ_min(H(x_t)) > −1e-3` | `δ ∈ {0.25, 0.5, 1.0}` |

All four are computed on the **same** set of trajectories per configuration,
so any ranking difference between criteria is attributable to the criterion
itself, not to different runs.

## Benchmark functions

Five standard non-convex functions, each with at least one verified saddle
point: Himmelblau, Ackley, Rastrigin, Styblinski-Tang, and Levy. Rosenbrock
is excluded (it is unimodal and contains no saddle point).

## Optimizers

- Fixed-rate gradient descent (`GD_fixed`)
- Adam (`β1=0.9, β2=0.999, ε=1e-8`)
- RMSProp (`α=0.99, ε=1e-8`)
- AdaGrad (`ε=1e-8`)

Each is swept across learning rates `{0.001, 0.01, 0.05, 0.1, 0.2, 0.5}`.

## Pipeline overview

1. **Saddle verification** (`find_saddles`) — for each function, a 240×240
   grid over the function domain is evaluated, the 800 lowest-gradient-norm
   points are used as candidates, and each is refined by solving
   `∇f(x) = 0` directly via `scipy.optimize.fsolve`. A candidate is accepted
   as a genuine saddle only if:
   - `‖∇f(x)‖ < 1e-6`
   - `λ_min(H(x)) < −1e-4`
   - `λ_max(H(x)) > 1e-4`

   Up to 3 verified saddles are kept per function, deduplicated by distance.

2. **Trajectory generation** (`run_config`) — for every `(function,
   optimizer, learning rate)` combination, 200 trials are run for 200 steps
   each, initialized as `x_0 ~ N(x_s, 0.1² I)` around the verified saddle.
   `torch.manual_seed(42)` is set at the start of every configuration for
   reproducibility.

3. **Criterion evaluation** — all four escape criteria, across all
   parameter variants (12 total: 3 per family), are checked at every step of
   every trajectory. The first step at which each criterion is satisfied is
   recorded per trial.

4. **SEE computation** (`see_pt`, `see_ci`) — for each `(function,
   optimizer, criterion, parameter)` combination, `SEE = P_esc /
   mean_escape_step` is computed, along with a 95% bootstrap confidence
   interval (2,000 resamples, percentile method).

5. **Cross-criterion analysis** — Spearman rank correlation between
   optimizer rankings under different criteria (both within-family, across
   parameter extremes, and between families, across criteria), and
   Kendall's W concordance across all four criteria simultaneously.

## Requirements

```
torch
numpy
pandas
scipy
matplotlib
```

## Usage

Run the main pipeline first:

```bash
python see_pipeline.py
```

This will:
- Print the verified saddle locations and curvature (`λ_min`, `r_curv`) for
  each function.
- Print SEE tables at fixed learning rate (0.2, with bootstrap CIs) and at
  each optimizer's best learning rate, per criterion.
- Print the best-performing optimizer per criterion, per function.
- Print pairwise Spearman correlations and Kendall's W concordance across
  the four criteria.
- Print within-family threshold stability (Spearman correlation between
  extreme parameter values of the same criterion family).
- Save `results_final/main_lr02.csv` and `results_final/best_lr.csv`.

Then generate the paper figures from the printed results:

```bash
python make_figures.py
```

This produces six figures in `figures/`:
- `fig1_criteria_grid.png` — best-LR SEE across all functions/criteria/optimizers.
- `fig2_spearman_heatmap.png` — pairwise Spearman correlation matrix between criteria.
- `fig3_kendall_w.png` — Kendall's W concordance per function.
- `fig4_B_vs_D.png` — criterion B vs. D scatter (geometry-based mutual validation).
- `fig5_A_vs_B.png` — criterion A vs. B scatter (distance vs. curvature ranking inversion).
- `fig6_lr02_with_CI.png` — SEE at fixed LR=0.2 with bootstrap error bars.

**Note:** `make_figures.py` currently contains hardcoded result values copied
from a completed run of `see_pipeline.py`, rather than reading the CSVs
directly. If you change any experimental parameters, re-run
`see_pipeline.py` first and manually update the `BEST`, `LR02`, `CI02`,
`SPEARMAN`, and `KENDALL_W` dictionaries at the top of `make_figures.py`
before regenerating figures, or refactor it to load from
`results_final/*.csv` directly.

## Known limitations

- All experiments are in 2D on classical benchmark functions; whether the
  same criterion-dependence holds on higher-dimensional loss surfaces (e.g.
  a real neural network) is untested here.
- Only four optimizers are evaluated; AdamW and SGD with momentum are not
  included.
- With only 4 optimizers, each individual Spearman/Kendall value is computed
  on 4 data points — read any single coefficient with appropriate caution.
  The paper leans on consistency of the pattern across 5 independent
  functions rather than any one coefficient.
- Criterion C's Spearman correlation is undefined (constant input) on
  Rastrigin, since every optimizer achieves SEE_C = 1.000 there. This is
  reported as "n/a" in the paper's concordance table and will raise a
  `ConstantInputWarning` from `scipy.stats.spearmanr` when reproducing this
  result — that warning is expected, not a bug.
- Exact bitwise reproducibility is only guaranteed on CPU; GPU execution may
  introduce minor non-determinism in Adam/RMSProp updates.

## Reproducing the paper's tables

Table I (verified saddles), Table II (best-LR SEE under all four criteria),
and Table III (Kendall's W and pairwise Spearman correlations) were all
regenerated directly from this code and match the values printed by
`see_pipeline.py` under the fixed seed (`SEED = 42`).

## Citation

If you use this code, please cite the accompanying paper (citation details
withheld here for double-blind review; see the paper PDF).
