# Saddle Escape Efficiency (SEE)

Code and results for the paper *Saddle Escape Efficiency: Introducing a Novel
Metric and Revealing Its Criterion-Dependence in Non-Convex Landscapes.*

This repository is laid out so a reviewer can read the source, check the
tables, and look at the figures without hunting through extra folders.

```
SEE = P_esc / mean_escape_step
```

Escape is scored four ways on the **same** trajectories. Rankings that
disagree are therefore from the criterion, not from different runs.

## Folder map

| Folder | What to open |
|---|---|
| `code/` | Source. Start with `core.py`, then `run_exp1.py`, `run_exp2.py`, `run_exp3.py`, `run_analysis.py`. |
| `results/1_two_dimensional/` | Experiment 1 tables and figures (8 functions, 6 optimizers, 4 criteria). |
| `results/2_higher_dimension/` | Experiment 2 (Rastrigin and Ackley at 10D and 50D). |
| `results/3_neural_network/` | Experiment 3 (XOR MLP saddle). |
| `results/run_log.txt` | Full printed output from the Tesla T4 run. |
| `writeup/notes.tex` | Tables, CIs, and figure captions for the August 2026 run. |
| `original_may2026/` | Original notebook (4 optimizers, 5 functions, 2D). |

`code/kaggle/` is the same pipeline as a single-file GPU script. It is not
needed to review the logic; `code/*.py` is the readable copy.

## How to run

From `code/`:

```bash
python smoke.py
python run_exp1.py
python run_exp2.py
python run_exp3.py
python run_analysis.py
```

The tables and figures already in `results/` are from the completed T4 run
(16 August 2026, ~77 minutes). Re-running is optional.

Needs: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib` (`requirements.txt`).
`SEED = 42`. Exact bitwise match is a CPU claim.

To compile the writeup: `cd writeup && pdflatex notes.tex`.

## What the code does

Four escape criteria, all on the same trajectories:

| | Definition | Sweep |
|---|---|---|
| A | `‖x_t − x_s‖ > r` | `r ∈ {1.5, 2.0, 3.0}` |
| B | `λ_min(H(x_t)) > −ε` | `ε ∈ {1e-2, 1e-3, 1e-4}` |
| C | `\|⟨x_t − x_s, v⟩\| > c · r_curv` | `c ∈ {0.5, 1.0, 2.0}`, `r_curv = 1/√\|λ_min\|` |
| D | `f(x_t) < f(x_s) − δ` and `λ_min > −1e-3` | `δ ∈ {0.25, 0.5, 1.0}` |

Headline parameters: `r=2`, `ε=1e-3`, `c=1`, `δ=0.5`.

Optimizers: fixed-rate GD, Adam, RMSProp, AdaGrad (`eps=1e-8`), AdamW
(`weight_decay=0.01`), SGD with momentum 0.9.

Rosenbrock is not in the suite (unimodal, no saddle). Booth is in the 2D
list but no verified saddle was found, so it is dropped from the tables.

## Main numbers (T4 run)

- A-vs-B rank inversion on 2D Ackley (`ρ = −0.81`) and Rastrigin (`−0.79`),
  and Rastrigin 10D (`−0.74`). Ackley loses the inversion with dimension
  (`−0.81 → +0.14 → +0.78`). XOR-MLP: `ρ = +0.81`.
- Rastrigin 50D: every optimizer has `best_A = 1`, so Spearman is undefined.
- Curvature-sharpness on the original five functions:
  `ρ(|λ_min|, W) = −0.7`, exact permutation `p = 0.2333` (n=5).
- Criterion C saturates on Himmelblau, Ackley, Rastrigin, Styblinski
  (4/6 optimizers at 1.0). It still splits Levy, Beale, and Schwefel.

Full tables and bootstrap CIs are in `writeup/notes.tex` and `results/`.
