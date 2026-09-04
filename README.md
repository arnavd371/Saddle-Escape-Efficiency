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
| `code/run_medal.py` | Mechanism, shared vs independent, A vs B selection. |
| `results/1_two_dimensional/` | T4 tables and figures (8 functions). |
| `results/2_higher_dimension/` | Rastrigin / Ackley at 10D and 50D. |
| `results/3_neural_network/` | XOR MLP. |
| `results/4_medal/` | Mechanism, protocol, and selection tables. |
| `writeup/notes.tex` | Numbers and CIs from the T4 run. |
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
python run_medal.py
```

`python algorithm.py` scores GD and Adam on Himmelblau (short check).
`results/1_two_dimensional/` through `3_neural_network/` are from the T4 run
(16 August 2026, ~77 min). `results/4_medal/` is the follow-up
(N=80, T=120, 4 functions).

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
- Test C saturates on Himmelblau, Ackley, Rastrigin, Styblinski
  (4/6 optimizers at 1.0). It still splits Levy, Beale, and Schwefel.

On Ackley, A ranks RMSProp first (`SEE_A = 1.00`) and AdaGrad last (`0.00`);
B ranks AdaGrad first (`SEE_B = 0.44`). On Rastrigin, A ranks RMSProp first
(`1.00`) and Adam last (`0.00`); B ranks Adam first (`0.71`).

## Follow-up (`results/4_medal/`)

N=80, T=120, lrs `{0.01, 0.1, 0.2}`, functions Himmelblau, Ackley, Rastrigin, Levy.

Mechanism: mean alignment with `v_min` at the first A-hit is high on
Himmelblau (`0.95`) and Levy (`0.97`), lower on Ackley (`0.67`) and
Rastrigin (`0.69`). The inversion functions leave off the unstable axis.

Shared vs independent paths:

| Function | rho shared | rho independent | Best-B flip |
|---|---|---|---|
| Himmelblau | +0.90 | +0.81 | RMSProp to SGD_mom |
| Ackley | -0.76 | -0.76 | no |
| Rastrigin | -0.75 | -0.79 | no |
| Levy | +0.23 | +0.26 | no |

Independent runs can change the B-winner on Himmelblau. The Ackley and
Rastrigin inversions stay.

Selection (argmax SEE_A vs argmax SEE_B):

| Function | Pick A | Pick B | Outcome |
|---|---|---|---|
| Rastrigin | RMSProp, f=16.9 | Adam, f=0.52 | B-pick ~30x lower loss |
| Ackley | RMSProp, 41% leave saddle | AdaGrad, 100% leave | A-pick often still has negative curvature |
| Levy | RMSProp lr=0.2, f=3.49 | RMSProp lr=0.1, f=0.44 | same trainer, B picks the better step size |
| Himmelblau | RMSProp | RMSProp | same pick |

A does not select the trainer that leaves the saddle. On Rastrigin the two
picks differ by a factor of about 30 in final f. On Ackley, AdaGrad's final
f is not better, so B is not a general loss minimizer.
