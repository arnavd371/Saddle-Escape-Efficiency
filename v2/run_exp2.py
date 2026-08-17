"""Exp 2 — Rastrigin / Ackley at d=10 and d=50. Criteria A and B only."""
import os, time
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core

OUTDIR, FIGDIR = 'results/out2', 'results/figs2'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

SEED, N, TMAX = 42, 100, 500
LRS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
FAMS = ['A', 'B']
t0 = time.time()

FUNCSND = {'Rastrigin': (core.f_rastrigin, 5.12), 'Ackley': (core.f_ackley, 5.0)}
DIMS = [10, 50]

def lambda_fn_nd(F, d):
    def f(Xd):
        lmin, _ = core.lanczos_extremes(F, Xd, k=min(30, d))
        return lmin
    return f

GEOM = {}
for name, (F, L) in FUNCSND.items():
    for d in DIMS:
        ss = core.find_saddles_nd(F, d, L, seed=SEED)
        if not ss:
            print(f'{name} d={d}: no saddle found (excluded)')
            continue
        s = torch.tensor(ss[0])
        lmin, lmax = core.lanczos_extremes(F, s[None], k=min(30, d))
        s = s.to(core.DEVICE)
        GEOM[(name, d)] = dict(s=s, lmin=lmin[0])
        print(f'{name} d={d}: {len(ss)} saddle(s), lmin={lmin[0]:.2f}')
print('saddle search', round(time.time() - t0, 1))

DATA = {}
for (name, d) in GEOM:
    F, L = FUNCSND[name]
    g, lf = GEOM[(name, d)], lambda_fn_nd(F, d)
    for o in core.OPTS:
        for lr in LRS:
            DATA[(name, d, o, lr)] = core.run_config(
                F, g['s'], None, None, None, o, lr, N, TMAX, SEED, FAMS, lf)
    print(name, d, 'done', round(time.time() - t0, 1))

best_rows = []
for (name, d) in GEOM:
    for o in core.OPTS:
        rec = {'function': name, 'dim': d, 'optimizer': o}
        for fam in FAMS:
            k = core.headline_idx(fam)
            rec[f'best_{fam}'] = max(
                core.see_pt(DATA[(name, d, o, lr)][0][k], DATA[(name, d, o, lr)][1][k]) for lr in LRS)
        best_rows.append(rec)
pd.DataFrame(best_rows).to_csv(f'{OUTDIR}/best_lr.csv', index=False)

rho_rows = []
for (name, d) in GEOM:
    sub = [r for r in best_rows if r['function'] == name and r['dim'] == d]
    a = [r['best_A'] for r in sub]
    b = [r['best_B'] for r in sub]
    rho = stats.spearmanr(a, b).correlation
    rho_rows.append({'function': name, 'dim': d, 'rho_AB': rho})
    print(name, d, 'rho(A,B)=', rho)

anchor = 'results/out1/spearman.csv'
if os.path.exists(anchor):
    sp1 = pd.read_csv(anchor)
    for name in FUNCSND:
        row = sp1[(sp1['function'] == name) & (sp1['pair'] == 'A-B')]
        if len(row):
            rho_rows.append({'function': name, 'dim': 2, 'rho_AB': row.iloc[0]['rho']})
pd.DataFrame(rho_rows).to_csv(f'{OUTDIR}/rho_vs_dim.csv', index=False)

fig, ax = plt.subplots(figsize=(5, 3.5))
for name, c in [('Rastrigin', '#4477aa'), ('Ackley', '#ee7733')]:
    sub = sorted([r for r in rho_rows if r['function'] == name], key=lambda r: r['dim'])
    if sub:
        ax.plot([r['dim'] for r in sub], [r['rho_AB'] for r in sub], 'o-', color=c, label=name)
ax.set_xscale('log')
ax.set_xlabel('dimension')
ax.set_ylabel('Spearman rho(A,B)')
ax.axhline(0, color='gray', linewidth=0.7)
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/rho_ab_vs_dim.png', dpi=300, bbox_inches='tight')
plt.close()
print('exp2 total', round(time.time() - t0, 1))
