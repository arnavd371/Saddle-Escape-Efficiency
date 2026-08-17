# v2 pipeline

From this directory:

```
python smoke.py
python run_exp1.py
python run_exp2.py
python run_exp3.py
python run_analysis.py
```

`results/` already has the T4 run (16 Aug 2026, kernel
`arnavd371/see-pipeline-v2-rebuild`, ~77 min). `notes.tex` is the writeup;
compile it here so the figure paths resolve.

`kaggle/main.py` is the same code inlined for a script kernel. Push with
Tesla T4, not the default P100.
