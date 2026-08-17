# Saddle Escape Efficiency (SEE)

SEE is an evaluation algorithm for first-order methods at saddles.
It scores several escape tests on one shared set of trajectories, then
returns a scalar per test:

```
SEE = P_esc / mean first hitting time
```

Because the paths are shared, a rank change across tests is from the
test, not from a different run.

Start at `code/algorithm.py`. Experiments in `run_exp*.py` are clients
of that file.

```
SEE-Eval(F, x_s, opts, lrs, N, T, C)
1. For each optimizer u and step size lr:
   a. Draw N particles:  X_0 = x_s + 0.1 * N(0, I)
   b. For t = 1 .. T:
        X_t <- u-step(X_{t-1})
        for each test c in C:
            if c has not hit and oracle_c(X_t) is true:
                tau_{i,c} <- t
   c. SEE_c(u, lr) = mean(escaped) / mean(tau | escaped)
2. best_c(u) = max_lr SEE_c(u, lr)
3. Return Spearman correlations and Kendall W of the best_* rankings
```

Oracles (headline parameters `r=2`, `eps=1e-3`, `c=1`, `delta=0.5`):

| | Test | Sweep |
|---|---|---|
| A | `||x_t - x_s|| > r` | `r in {1.5, 2.0, 3.0}` |
| B | `lambda_min(H(x_t)) > -eps` | `eps in {1e-2, 1e-3, 1e-4}` |
| C | `|<x_t - x_s, v>| > c * r_curv` | `c in {0.5, 1.0, 2.0}`, `r_curv = 1/sqrt(|lambda_min|)` |
| D | `f(x_t) < f(x_s) - delta` and `lambda_min > -1e-3` | `delta in {0.25, 0.5, 1.0}` |

Call:

```python
import algorithm as see
out, esc, stp = see.evaluate(F, s, v, r_curv, f_s, 'Adam', 0.2,
                             N, Tmax, seed, ['A', 'B'], lambda_fn, rng)
```

`see.simulate` is the inner loop. `see.oracle` is one test. `see.see`
and `see.see_ci` turn hitting times into the scalar. `see.rank_agreement`
is step 3.

## Folder map

| Folder | What to open |
|---|---|
| `code/algorithm.py` | The algorithm. |
| `code/core.py` | Losses, saddles, Hessians, optimizer constructors. |
| `code/run_exp1.py` | 2D clients (8 functions, 6 optimizers, 4 tests). |
| `code/run_exp2.py` | 10D / 50D clients. |
| `code/run_exp3.py` | XOR MLP client. |
| `results/` | Tables and figures from the Tesla T4 run. |
| `writeup/notes.tex` | Numbers and CIs. |
| `original_may2026/` | Earlier notebook. |

## How to run

From `code/`:

```bash
python algorithm.py
python smoke.py
python run_exp1.py
python run_exp2.py
python run_exp3.py
python run_analysis.py
```

`python algorithm.py` scores GD and Adam on Himmelblau (short check).
The tables in `results/` are from the T4 run (16 August 2026, ~77 min).

Needs: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib` (`requirements.txt`).
`SEED = 42`.

Optimizers: GD, Adam, RMSProp, AdaGrad (`eps=1e-8`), AdamW
(`weight_decay=0.01`), SGD momentum 0.9.

Rosenbrock is not in the suite (unimodal, no saddle). Booth has no
verified saddle and is dropped from the tables.

## Numbers from the T4 run

- A-vs-B rank inversion on 2D Ackley (`rho = -0.81`) and Rastrigin (`-0.79`),
  and Rastrigin 10D (`-0.74`). Ackley loses the inversion with dimension
  (`-0.81 -> +0.14 -> +0.78`). XOR-MLP: `rho = +0.81`.
- Rastrigin 50D: every optimizer has `best_A = 1`, so Spearman is undefined.
- Curvature-sharpness on the original five functions:
  `rho(|lambda_min|, W) = -0.7`, exact permutation `p = 0.2333` (n=5).
- Test C saturates on Himmelblau, Ackley, Rastrigin, Styblinski
  (4/6 optimizers at 1.0). It still splits Levy, Beale, and Schwefel.
