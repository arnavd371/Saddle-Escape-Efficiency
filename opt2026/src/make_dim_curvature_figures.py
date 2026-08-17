"""
New Table/Figure outputs for the dimensionality sweep (Ext 2) and continuous
curvature-sharpness sweep (Ext 3). Reads exclusively from committed
results/*.csv -- no simulation, no Kaggle.
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

DIM_FUNCS = ['Ackley', 'Rastrigin', 'Styblinski']
COLORS = {'Ackley': '#4477aa', 'Rastrigin': '#ee7733', 'Styblinski': '#228833'}
MARKERS = {'Ackley': 'o', 'Rastrigin': 's', 'Styblinski': '^'}

# ============================================================ Table V: dimensionality summary
print('Writing Table V (dimensionality sweep summary)...')
dfs = []
for fn in DIM_FUNCS:
    df = pd.read_csv(os.path.join(RESULTS, f'ext2_{fn}_spearman_vs_dim.csv'))
    dfs.append(df)
df_dim = pd.concat(dfs, ignore_index=True).sort_values(['function', 'n'])
df_dim.round(3).to_csv(os.path.join(RESULTS, 'table5_dimensionality_summary.csv'), index=False)
with open(os.path.join(RESULTS, 'table5_dimensionality_summary.md'), 'w') as f:
    f.write('# Table V: A-B and B-D Spearman rho vs. dimension (n=7 optimizers)\n\n')
    f.write('| Function | n (dim) | A-B rho | B-D rho | \\|A-B rho\\| | 1 - B-D rho |\n')
    f.write('|---|---|---|---|---|---|\n')
    for _, r in df_dim.iterrows():
        f.write(f"| {r['function']} | {int(r['n'])} | {r['rho_AB']:+.3f} | {r['rho_BD']:+.3f} | "
                 f"{r['abs_rho_AB']:.3f} | {r['one_minus_rho_BD']:.3f} |\n")

# ============================================================ Figure 7: |A-B rho| and |1-B-D rho| vs dimension
print('Rendering Figure 7 (dimensionality sweep: |A-B rho| and |1-B-D rho| vs dimension)...')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for fn in DIM_FUNCS:
    sub = df_dim[df_dim['function'] == fn].sort_values('n')
    axes[0].plot(sub['n'], sub['abs_rho_AB'], marker=MARKERS[fn], color=COLORS[fn], label=fn, linewidth=1.5)
    axes[1].plot(sub['n'], sub['one_minus_rho_BD'], marker=MARKERS[fn], color=COLORS[fn], label=fn, linewidth=1.5)
axes[0].set_xscale('log'); axes[1].set_xscale('log')
axes[0].set_xlabel('Dimension n (log scale)'); axes[0].set_ylabel('|A-B rho|')
axes[0].set_title('Distance vs. curvature divergence')
axes[0].set_ylim(-0.05, 1.05)
axes[1].set_xlabel('Dimension n (log scale)'); axes[1].set_ylabel('1 - B-D rho')
axes[1].set_title('Curvature-exit vs. loss-drop disagreement')
axes[1].set_ylim(-0.05, 1.05)
for ax in axes:
    ax.set_xticks([2, 5, 10, 25, 50]); ax.set_xticklabels(['2', '5', '10', '25', '50'])
    ax.legend(fontsize=8)
plt.suptitle("Testing the paper's dimensionality hypothesis (n=7 optimizers)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'fig7_dimensionality_sweep.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================ Table VI + Figure 8: curvature sweep
print('Writing Table VI and rendering Figure 8 (continuous curvature sweep: W vs k)...')
df_w_k = pd.read_csv(os.path.join(RESULTS, 'curvature_kendall_w_vs_k.csv'))
df_corr = pd.read_csv(os.path.join(RESULTS, 'curvature_k_vs_W_correlation.csv'))
df_w_k.round(3).to_csv(os.path.join(RESULTS, 'table6_curvature_summary.csv'), index=False)
with open(os.path.join(RESULTS, 'table6_curvature_summary.md'), 'w') as f:
    f.write("# Table VI: Kendall's W vs. curvature-sharpness parameter k (n=7 optimizers)\n\n")
    f.write('| k | W | 95% CI |\n|---|---|---|\n')
    for _, r in df_w_k.iterrows():
        f.write(f"| {r['k']:.3f} | {r['W']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |\n")
    r = df_corr.iloc[0]
    f.write(f"\nSpearman rho(k, W) = {r['rho']:+.3f}, n={int(r['n'])} k-values, "
             f"95% bootstrap CI = [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] (B={int(r['B'])} resamples).\n")
    f.write("\n**This CI is wide primarily because each individual W(k) estimate is itself noisy "
             "at n_items=7 optimizers (see per-k CIs above, each ~0.35-0.5 units wide), not because "
             "the k-sweep (n=18 conditions) is too coarse** -- see SANITY_CHECKS.md Check 2 for detail. "
             "Read as underpowered-to-detect-a-relationship, not as evidence of no relationship.\n")

fig, ax = plt.subplots(figsize=(7, 4))
ks = df_w_k['k'].values
Ws = df_w_k['W'].values
los = df_w_k['ci_lo'].values
his = df_w_k['ci_hi'].values
ax.fill_between(ks, los, his, alpha=0.2, color='#4477aa', label='95% bootstrap CI')
ax.plot(ks, Ws, marker='o', color='#4477aa', linewidth=1.5, label="Kendall's W")
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Curvature-sharpness parameter k (0 = smooth quadratic saddle)')
ax.set_ylabel("Kendall's W (concordance across 4 criteria)")
r = df_corr.iloc[0]
ax.set_title(f"Continuous curvature sweep: W vs. k (n=7 optimizers)\n"
              f"Spearman rho(k,W)={r['rho']:+.2f}, 95% CI [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
              f"(n={int(r['n'])} k-values) -- CI crosses zero, underpowered per-W-estimate not per-k-count",
              fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'fig8_curvature_sweep.png'), dpi=300, bbox_inches='tight')
plt.close()

print('\nTable V/VI and Figures 7-8 written to results/.')
