"""Exp 3 — XOR MLP saddle, criteria A and B."""
import os, time
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core
from nn import xor_data, make_loss_fn, D_PARAM

OUTDIR, FIGDIR = 'results/out3', 'results/figs3'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

SEED, N, TMAX = 42, 50, 300
LRS = [0.001, 0.005, 0.01, 0.05, 0.1]
FAMS = ['A', 'B']
DOM_L = 4.0  # covers the saturated-tanh region where random inits land
t0 = time.time()

X, y = xor_data(seed=SEED)
F_cpu = make_loss_fn(X, y)
F = make_loss_fn(X.to(core.DEVICE), y.to(core.DEVICE))

ss = core.find_saddles_nd(F_cpu, D_PARAM, DOM_L, seed=SEED)
print('saddles found', len(ss), 'time', round(time.time() - t0, 1))
s = torch.tensor(ss[0])
lmin, lmax = core.lanczos_extremes(F_cpu, s[None], k=25)
print('using saddle, loss=', F_cpu(s[None]).item(), 'lmin=', lmin[0], 'lmax=', lmax[0])
s = s.to(core.DEVICE)

def lambda_fn(Xd):
    lm, _ = core.lanczos_extremes(F, Xd, k=25)
    return lm

DATA = {}
for o in core.OPTS:
    for lr in LRS:
        DATA[(o, lr)] = core.run_config(F, s, None, None, None, o, lr, N, TMAX, SEED, FAMS, lambda_fn)
    print(o, 'done', round(time.time() - t0, 1))

best_rows = []
for o in core.OPTS:
    rec = {'optimizer': o}
    for fam in FAMS:
        k = core.headline_idx(fam)
        rec[f'best_{fam}'] = max(
            core.see_pt(DATA[(o, lr)][0][k], DATA[(o, lr)][1][k]) for lr in LRS)
    best_rows.append(rec)
pd.DataFrame(best_rows).to_csv(f'{OUTDIR}/best_lr.csv', index=False)

a = [r['best_A'] for r in best_rows]
b = [r['best_B'] for r in best_rows]
rho = stats.spearmanr(a, b).correlation
print('NN rho(A,B)=', rho)
pd.DataFrame([{'rho_AB': rho, 'lmin_saddle': lmin[0]}]).to_csv(f'{OUTDIR}/rho_ab.csv', index=False)
for r in best_rows:
    print(f"{r['optimizer']:9s} A:{r['best_A']:.3f} B:{r['best_B']:.3f}")

fig, ax = plt.subplots(figsize=(5, 3.5))
xw = np.arange(len(core.OPTS))
ax.bar(xw - 0.15, [r['best_A'] for r in best_rows], width=0.3, label='A (distance)', color='#4477aa')
ax.bar(xw + 0.15, [r['best_B'] for r in best_rows], width=0.3, label='B (curvature)', color='#ee7733')
ax.set_xticks(xw)
ax.set_xticklabels(core.OPTS, rotation=45, fontsize=8)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title(f'XOR-MLP saddle, rho(A,B)={rho:+.2f}')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/nn_AB.png', dpi=300, bbox_inches='tight')
plt.close()
print('exp3 total', round(time.time() - t0, 1))
