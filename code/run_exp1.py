import os, time
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core
import algorithm as see

OUTDIR, FIGDIR = '../results/1_two_dimensional/tables', '../results/1_two_dimensional/figures'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

SEED, N, TMAX = 42, 200, 200
LRS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
FAMS = ['A', 'B', 'C', 'D']
rng = np.random.default_rng(SEED)
t0 = time.time()

def lambda_fn_2d(F):
    def f(Xd):
        H = core.hessian(F, Xd)
        return core.safe_eigh(H, eigenvectors=False).cpu().numpy()[:, 0]
    return f

GEOM = {}
for name, (F, L) in core.FUNCS2D.items():
    ss = core.find_saddles_2d(F, L)
    if not ss:
        print(f'{name}: no saddle found (excluded)')
        continue
    s = torch.tensor(ss[0])
    lmin, lmax, v = core.eigh_min(F, s[None])
    lmin = lmin.item()
    v = v[0] / v[0].norm()
    r_curv = 1 / np.sqrt(abs(lmin))
    f_s = F(s[None])[0].item()
    s, v = s.to(core.DEVICE), v.to(core.DEVICE)
    GEOM[name] = dict(s=s, v=v, r_curv=r_curv, f_s=f_s, lmin=lmin, n_saddles=len(ss))
    print(f'{name}: {len(ss)} saddle(s), s={s.cpu().numpy().round(3)}, lmin={lmin:.2f}, r_curv={r_curv:.3f}')
print('saddle search', round(time.time() - t0, 1))

DATA = {}
for name in GEOM:
    F, L = core.FUNCS2D[name]
    g, lf = GEOM[name], lambda_fn_2d(F)
    for o in core.OPTS:
        for lr in LRS:
            DATA[(name, o, lr)] = see.simulate(
                F, g['s'], g['v'], g['r_curv'], g['f_s'], o, lr, N, TMAX, SEED, FAMS, lf)
    print(name, 'done', round(time.time() - t0, 1))

rows = []
for name in GEOM:
    for o in core.OPTS:
        e, s = DATA[(name, o, 0.2)]
        rec = {'function': name, 'optimizer': o}
        for fam in FAMS:
            k = see.headline_idx(fam)
            val, lo, hi = see.see_ci(e[k], s[k], rng)
            rec[f'SEE_{fam}'] = val
            rec[f'CI_lo_{fam}'] = lo
            rec[f'CI_hi_{fam}'] = hi
        rows.append(rec)
pd.DataFrame(rows).to_csv(f'{OUTDIR}/main_lr02.csv', index=False)

best_rows = []
for name in GEOM:
    for o in core.OPTS:
        rec = {'function': name, 'optimizer': o}
        for fam in FAMS:
            k = see.headline_idx(fam)
            rec[f'best_{fam}'] = max(
                see.see(DATA[(name, o, lr)][0][k], DATA[(name, o, lr)][1][k]) for lr in LRS)
        best_rows.append(rec)
pd.DataFrame(best_rows).to_csv(f'{OUTDIR}/best_lr.csv', index=False)

pairs = [(a, b) for i, a in enumerate(FAMS) for b in FAMS[i + 1:]]
spear_rows, kw_rows = [], []
for name in GEOM:
    sub = [r for r in best_rows if r['function'] == name]
    mat = np.array([[r[f'best_{f}'] for f in FAMS] for r in sub])
    for a, b in pairs:
        ia, ib = FAMS.index(a), FAMS.index(b)
        rho = stats.spearmanr(mat[:, ia], mat[:, ib]).correlation
        spear_rows.append({'function': name, 'pair': f'{a}-{b}', 'rho': rho})
    kw_rows.append({'function': name, 'W': see.kendall_w(mat.T), 'lmin': GEOM[name]['lmin']})
pd.DataFrame(spear_rows).to_csv(f'{OUTDIR}/spearman.csv', index=False)
pd.DataFrame(kw_rows).to_csv(f'{OUTDIR}/kendall_w.csv', index=False)

stab_rows = []
for fam, ps in [('A', [1.5, 2.0, 3.0]), ('B', [1e-2, 1e-3, 1e-4]),
                ('C', [0.5, 1.0, 2.0]), ('D', [0.25, 0.5, 1.0])]:
    for name in GEOM:
        rank = {}
        for p in ps:
            k = see.VAR.index((fam, p))
            rank[p] = [max(see.see(DATA[(name, o, lr)][0][k], DATA[(name, o, lr)][1][k])
                           for lr in LRS) for o in core.OPTS]
        rho = stats.spearmanr(rank[ps[0]], rank[ps[-1]]).correlation
        stab_rows.append({'family': fam, 'function': name, 'rho': rho})
pd.DataFrame(stab_rows).to_csv(f'{OUTDIR}/within_family.csv', index=False)

print('printed tables')
for name in GEOM:
    print(name)
    for o in core.OPTS:
        r = [x for x in best_rows if x['function'] == name and x['optimizer'] == o][0]
        print(f"  {o:9s} " + ' '.join(f'{f}:{r[f"best_{f}"]:.3f}' for f in FAMS))

names = list(GEOM.keys())
fig, axes = plt.subplots(len(names), 4, figsize=(14, 2.1 * len(names)), squeeze=False)
crit_names = {'A': 'fixed radius', 'B': 'curvature', 'C': 'eigen-disp', 'D': 'loss-drop'}
colors = dict(zip(core.OPTS, ['#4477aa', '#ee7733', '#228833', '#cc3311', '#aa3377', '#66ccee']))
for i, name in enumerate(names):
    sub = [r for r in best_rows if r['function'] == name]
    for j, fam in enumerate(FAMS):
        ax = axes[i, j]
        vals = [r[f'best_{fam}'] for r in sub]
        ax.bar(range(len(core.OPTS)), vals, color=[colors[o] for o in core.OPTS])
        ax.set_xticks(range(len(core.OPTS)))
        ax.set_xticklabels(core.OPTS, fontsize=6, rotation=45)
        ax.set_ylim(0, 1.05)
        if j == 0:
            ax.set_ylabel(name, fontsize=8)
        if i == 0:
            ax.set_title(crit_names[fam], fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig1_criteria_grid.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for idx, name in enumerate(names):
    ax = axes[idx // 4, idx % 4]
    M = np.eye(4)
    for a, b in pairs:
        r = [x for x in spear_rows if x['function'] == name and x['pair'] == f'{a}-{b}'][0]['rho']
        ia, ib = FAMS.index(a), FAMS.index(b)
        M[ia, ib] = M[ib, ia] = r
    ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(FAMS)
    ax.set_yticklabels(FAMS)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f'{M[a, b]:+.2f}', ha='center', va='center', fontsize=7)
    ax.set_title(name, fontsize=9)
for idx in range(len(names), 8):
    axes[idx // 4, idx % 4].axis('off')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig2_spearman.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.barh(names, [r['W'] for r in kw_rows], color='#4477aa')
ax.set_xlim(0, 1)
ax.set_xlabel("Kendall's W")
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig3_kendall_w.png', dpi=300, bbox_inches='tight')
plt.close()
print('exp1 total', round(time.time() - t0, 1), 's')
