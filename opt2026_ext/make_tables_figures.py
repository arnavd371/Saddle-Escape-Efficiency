"""
Regenerates Tables II-IV and Figures 1-3 from committed results/*.csv, plus
new Table/Figure outputs for the dimensionality sweep (Ext 2) and continuous
curvature sweep (Ext 3).

Unlike the baseline notebook's Cell 2 (which works off hand-copied,
hardcoded dicts -- a known wart flagged in the baseline README itself), this
reads exclusively from the CSVs already written by the Kaggle kernels and
committed to results/. Change the underlying data, rerun this script, done --
no manual copy-paste step.

Table numbering follows the baseline README's own scheme (Table I = verified
saddles, Table II = best-LR SEE) plus the extension prompt's explicit
reference to "Table IV" for pairwise Spearman rho at best LR:
  Table II  = best-LR SEE per function/optimizer/criterion (n=7 optimizers)
  Table III = Kendall's W concordance per function, with bootstrap 95% CI
  Table IV  = pairwise Spearman correlations between criteria at best LR

This script only reads from disk and writes to results/ -- no simulation,
no Kaggle, nothing stochastic beyond what's already fixed in the saved CSVs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')

FUNCS = ['Himmelblau', 'Ackley', 'Rastrigin', 'Styblinski', 'Levy']
OPTS = ['GD_fixed', 'Adam', 'RMSProp', 'AdaGrad', 'AdamW', 'SGD_Nesterov', 'Lion']
FAMS = ['A', 'B', 'C', 'D']
CRIT_NAMES = {'A': 'A: fixed radius', 'B': 'B: curvature', 'C': 'C: eigen-disp', 'D': 'D: loss-drop'}
OPT_SHORT = {'GD_fixed': 'GD', 'Adam': 'Ad', 'RMSProp': 'RM', 'AdaGrad': 'AG',
             'AdamW': 'AdW', 'SGD_Nesterov': 'SGDN', 'Lion': 'Lion'}
COLORS = {'GD_fixed': '#4477aa', 'Adam': '#ee7733', 'RMSProp': '#228833', 'AdaGrad': '#cc3311',
          'AdamW': '#aa4488', 'SGD_Nesterov': '#66ccee', 'Lion': '#bbbb33'}

df_best = pd.read_csv(os.path.join(RESULTS, 'best_lr_ext.csv'))
df_lr02 = pd.read_csv(os.path.join(RESULTS, 'main_lr02_ext.csv'))
df_w = pd.read_csv(os.path.join(RESULTS, 'kendall_w_ext.csv'))
df_sp = pd.read_csv(os.path.join(RESULTS, 'spearman_pairwise_ext.csv'))
df_stab = pd.read_csv(os.path.join(RESULTS, 'within_family_stability_ext.csv'))

# ============================================================ Table II: best-LR SEE
print('Writing Table II (best-LR SEE, n=7 optimizers)...')
df_best.round(3).to_csv(os.path.join(RESULTS, 'table2_best_lr_see.csv'), index=False)
with open(os.path.join(RESULTS, 'table2_best_lr_see.md'), 'w') as f:
    f.write('# Table II: SEE at each optimizer\'s best LR, n=7 optimizers\n\n')
    for fn in FUNCS:
        sub = df_best[df_best['function'] == fn]
        f.write(f'## {fn}\n\n')
        f.write('| Optimizer | A (radius) | B (curvature) | C (eigen-disp) | D (loss-drop) |\n')
        f.write('|---|---|---|---|---|\n')
        for _, r in sub.iterrows():
            f.write(f"| {r['optimizer']} | {r['best_A']:.3f} | {r['best_B']:.3f} | {r['best_C']:.3f} | {r['best_D']:.3f} |\n")
        f.write('\n')

# ============================================================ Table III: Kendall's W with CI
print("Writing Table III (Kendall's W with bootstrap 95% CI, n=7 optimizers)...")
df_w_out = df_w[['function', 'W', 'n_items', 'n_judges', 'ci_lo', 'ci_hi']].round(3)
df_w_out.to_csv(os.path.join(RESULTS, 'table3_kendall_w.csv'), index=False)
with open(os.path.join(RESULTS, 'table3_kendall_w.md'), 'w') as f:
    f.write("# Table III: Kendall's W concordance across 4 criteria, n=7 optimizers\n\n")
    f.write('| Function | W | 95% CI | n_items (optimizers) | n_judges (criteria) |\n')
    f.write('|---|---|---|---|---|\n')
    for _, r in df_w_out.iterrows():
        f.write(f"| {r['function']} | {r['W']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {int(r['n_items'])} | {int(r['n_judges'])} |\n")

# ============================================================ Table IV: pairwise Spearman at best LR
print('Writing Table IV (pairwise Spearman correlations at best LR, n=7 optimizers)...')
df_sp_out = df_sp.round(3)
df_sp_out.to_csv(os.path.join(RESULTS, 'table4_pairwise_spearman.csv'), index=False)
with open(os.path.join(RESULTS, 'table4_pairwise_spearman.md'), 'w') as f:
    f.write('# Table IV: Pairwise Spearman rho between criteria at best LR, n=7 optimizers\n\n')
    f.write('| Function | ' + ' | '.join(sorted(df_sp['pair'].unique())) + ' |\n')
    f.write('|---|' + '---|' * len(df_sp['pair'].unique()) + '\n')
    for fn in FUNCS:
        sub = df_sp[df_sp['function'] == fn].set_index('pair')['rho']
        row = ' | '.join(f'{sub.get(p, float("nan")):+.2f}' for p in sorted(df_sp['pair'].unique()))
        f.write(f'| {fn} | {row} |\n')

# ============================================================ Fig 1: best-LR SEE grid (n=7 optimizers)
print('Rendering Figure 1 (best-LR SEE grid, n=7 optimizers)...')
fig, axes = plt.subplots(len(FUNCS), 4, figsize=(14, 2.1 * len(FUNCS)), squeeze=False)
for i, fn in enumerate(FUNCS):
    sub = df_best[df_best['function'] == fn].set_index('optimizer')
    for j, fam in enumerate(FAMS):
        ax = axes[i, j]
        vals = [sub.loc[o, f'best_{fam}'] for o in OPTS]
        bars = ax.bar(range(len(OPTS)), vals, color=[COLORS[o] for o in OPTS], width=0.7)
        ax.set_xticks(range(len(OPTS)))
        ax.set_xticklabels([OPT_SHORT[o] for o in OPTS], fontsize=6, rotation=45)
        ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1.0])
        if j == 0:
            ax.set_ylabel(fn, fontsize=8)
        else:
            ax.set_yticklabels([])
        if i == 0:
            ax.set_title(CRIT_NAMES[fam], fontsize=8)
plt.suptitle('Best-LR SEE under four escape criteria (n=7 optimizers)', fontsize=10, y=1.005)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'fig1_criteria_grid_ext.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================ Fig 2: Spearman heatmap
print('Rendering Figure 2 (pairwise Spearman heatmap, n=7 optimizers)...')
pairs = sorted(df_sp['pair'].unique())
M = np.full((len(FUNCS), len(pairs)), np.nan)
for i, fn in enumerate(FUNCS):
    sub = df_sp[df_sp['function'] == fn].set_index('pair')['rho']
    for j, p in enumerate(pairs):
        if p in sub.index:
            M[i, j] = sub[p]
fig, ax = plt.subplots(figsize=(6, 3.4))
im = ax.imshow(np.nan_to_num(M, nan=0.0), vmin=-1, vmax=1, cmap='RdBu', aspect='auto')
ax.set_xticks(range(len(pairs))); ax.set_xticklabels(pairs, fontsize=9)
ax.set_yticks(range(len(FUNCS))); ax.set_yticklabels(FUNCS, fontsize=9)
for i in range(len(FUNCS)):
    for j in range(len(pairs)):
        v = M[i, j]
        txt = 'n/a' if np.isnan(v) else f'{v:+.2f}'
        col = 'white' if (not np.isnan(v) and abs(v) > 0.6) else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, color=col)
plt.colorbar(im, fraction=0.03, pad=0.03)
ax.set_title('Pairwise Spearman $\\rho$ between criteria (best-LR ranks, n=7)', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'fig2_spearman_heatmap_ext.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================ Fig 3: Kendall's W with CI error bars
print("Rendering Figure 3 (Kendall's W with bootstrap CI, n=7 optimizers)...")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
Ws = [df_w[df_w['function'] == fn]['W'].values[0] for fn in FUNCS]
los = [df_w[df_w['function'] == fn]['ci_lo'].values[0] for fn in FUNCS]
his = [df_w[df_w['function'] == fn]['ci_hi'].values[0] for fn in FUNCS]
err_lo = [w - lo for w, lo in zip(Ws, los)]
err_hi = [hi - w for w, hi in zip(Ws, his)]
bars = ax.barh(FUNCS, Ws, color='#4477aa', alpha=0.85, height=0.5,
                xerr=[err_lo, err_hi], capsize=3, error_kw={'linewidth': 1})
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, label='W=0.5')
ax.axvline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlim(0, 1.32)
ax.set_xlabel("Kendall's $W$ (n=7 optimizers), 95% bootstrap CI")
ax.set_title('Cross-criterion concordance per function', fontsize=9)
for bar, w in zip(bars, Ws):
    ax.text(min(w + 0.03, 1.05), bar.get_y() + bar.get_height() / 2, f'{w:.2f}', va='center', fontsize=8)
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'fig3_kendall_w_ext.png'), dpi=300, bbox_inches='tight')
plt.close()

print('\nTables II-IV and Figures 1-3 (n=7 optimizers) written to results/.')
