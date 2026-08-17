# code/

Read in this order:

1. `core.py` — functions, saddles, optimizers, criteria, SEE
2. `run_exp1.py` — 2D benchmarks
3. `run_exp2.py` — 10D / 50D
4. `nn.py` then `run_exp3.py` — XOR MLP
5. `run_analysis.py` — printed summary
6. `smoke.py` — short check, not the full experiment

Run these from this directory. Output goes to `../results/`.

`kaggle/main.py` is the same pipeline inlined for a GPU script kernel.
Review `core.py` and the `run_exp*.py` files instead.
