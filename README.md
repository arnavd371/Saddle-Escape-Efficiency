# Saddle Escape Efficiency (SEE)

Code for *Saddle Escape Efficiency: Introducing a Novel Metric and Revealing
Its Criterion-Dependence in Non-Convex Landscapes.*

SEE is not one number. Its value, and even the optimizer ranking it implies,
depends on how "escape" is defined. This repo evaluates four criteria on the
**same** trajectories so that ranking differences come from the criterion,
not from different runs.

```
SEE = P_esc / mean_escape_step
```

Three snapshots live here. Nothing from the earlier work was deleted.

| Folder | When | What |
|---|---|---|
| `may2026/` | original paper code | 4 optimizers, 5 functions, 2D. Colab notebook. |
| `opt2026/` | summer 2026 extensions | 7 optimizers, dim sweep, curvature sweep, XOR net, CSVs and figures. |
| `v2/` | August 2026 | rewrite: 6 optimizers, 8 functions, 10D/50D, XOR MLP. T4 results included. |

The original notebook is also at the repo root (`Saddle_EE_Code.ipynb`) so
old links keep working.

## Criteria

| | Definition | Sweep |
|---|---|---|
| A | `‖x_t − x_s‖ > r` | `r ∈ {1.5, 2.0, 3.0}` |
| B | `λ_min(H(x_t)) > −ε` | `ε ∈ {1e-2, 1e-3, 1e-4}` |
| C | `\|⟨x_t − x_s, v⟩\| > c · r_curv` | `c ∈ {0.5, 1.0, 2.0}`, `r_curv = 1/√|λ_min|` |
| D | `f(x_t) < f(x_s) − δ` and `λ_min > −1e-3` | `δ ∈ {0.25, 0.5, 1.0}` |

Headline parameters: `r=2`, `ε=1e-3`, `c=1`, `δ=0.5`. C uses `r_curv`
because that is the length of the unstable manifold at a quadratic saddle;
a fixed radius is the wrong scale when `|λ_min|` spans two orders of
magnitude.

Rosenbrock is not in the suite. It is unimodal. There is no saddle to escape.

## v2 (August 2026)

Run from `v2/`:

```bash
python smoke.py          # cheap check
python run_exp1.py       # 2D, ~7 min on a T4
python run_exp2.py       # 10D/50D, ~1 hour
python run_exp3.py       # XOR MLP
python run_analysis.py
```

Full-scale numbers and 300 dpi figures from the T4 run are already in
`v2/results/`. Writeup: `v2/notes.tex` (compile next to `v2/results/figs*`).

Kaggle script kernel: `v2/kaggle/`. Needs Tesla T4. The default P100 is
sm_60 and will not run the preinstalled PyTorch.

What changed vs the original notebook:

- AdamW and SGD+momentum added (AdaGrad `eps` set to `1e-8`, not the
  PyTorch default `1e-10`).
- Beale, Booth, Schwefel added. Booth has no verified saddle and is dropped.
- Escape direction is `torch.linalg.eigh`, smallest-eigenvalue column,
  then unit-normalized. No 2x2 closed form.
- Saddle pick is deterministic: sort by gradient norm, then lexicographic
  coordinates, so CPU and GPU agree.
- Higher-d saddles: 5000 random inits, Lanczos instead of a dense Hessian.
- Bootstrap CIs are percentile `[lo, hi]`, 2000 resamples, not ± sigma.
- Diverged points clamped at `1e4` before curvature queries (`1e6` blows
  up cuSOLVER). `eigh` goes through `safe_eigh`.

August 2026 T4 headline:

- A-vs-B inversion holds on 2D Ackley (`ρ = −0.81`) and Rastrigin (`−0.79`),
  and on Rastrigin 10D (`−0.74`). It disappears on Ackley as dimension
  grows (`−0.81 → +0.14 → +0.78`). XOR-MLP: `ρ = +0.81`, no inversion.
- Rastrigin 50D: every optimizer has `best_A = 1`, so Spearman is undefined.
- Curvature–sharpness on the original five: `ρ(|λ_min|, W) = −0.7`, exact
  permutation `p = 0.2333`. Not significant at n=5.
- Criterion C saturates on Himmelblau / Ackley / Rastrigin / Styblinski
  (4/6 at 1.0). It still discriminates on Levy, Beale, and Schwefel.

## may2026

```bash
# Colab: open may2026/Saddle_EE_Code.ipynb
# or the copy at repo root
```

Four optimizers (GD, Adam, RMSProp, AdaGrad), five functions, 2D only.
See `may2026/README.md`.

## opt2026

Dimensionality sweep, extra optimizers (incl. Lion), curvature family,
XOR construction notes, and the CSVs/figures from those Kaggle runs.
Start at `opt2026/RESULTS_SUMMARY.md`.

## Requirements

```
torch
numpy
pandas
scipy
matplotlib
```

`SEED = 42`. Bitwise reproducibility is a CPU claim; GPU Adam/RMSProp
can drift a little.

## Citation

Citation details withheld here for double-blind review.
