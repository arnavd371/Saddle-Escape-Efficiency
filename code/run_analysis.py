"""Cross-experiment summary. Reads CSVs written by run_exp1/2/3."""
import itertools
import numpy as np
import pandas as pd
from scipy import stats

ORIGINAL5 = ['Himmelblau', 'Ackley', 'Rastrigin', 'Styblinski', 'Levy']
NEW3 = ['Beale', 'Booth', 'Schwefel']

def exact_spearman_p(x, y):
    # n=5: enumerate the permutation null rather than scipy's t approximation.
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rho0 = stats.spearmanr(x, y).correlation
    count = total = 0
    for perm in itertools.permutations(range(n)):
        r = stats.spearmanr(x, y[list(perm)]).correlation
        if abs(r) >= abs(rho0) - 1e-12:
            count += 1
        total += 1
    return rho0, count / total

kw = pd.read_csv('../results/1_two_dimensional/tables/kendall_w.csv').set_index('function')
sub = kw.loc[ORIGINAL5]
rho, p = exact_spearman_p(sub['lmin'].abs().values, sub['W'].values)
print('curvature-sharpness: rho(|lmin|,W) =', round(rho, 3), 'exact p =', round(p, 4), 'n=5')
sub.to_csv('../results/1_two_dimensional/tables/curvature_sharpness_input.csv')

print()
print('A-vs-B rank inversion across scales')
sp1 = pd.read_csv('../results/1_two_dimensional/tables/spearman.csv')
for name in ORIGINAL5 + NEW3:
    row = sp1[(sp1['function'] == name) & (sp1['pair'] == 'A-B')]
    if len(row):
        print(f'  2D {name:11s} rho(A,B)={row.iloc[0]["rho"]:+.2f}')

try:
    r2 = pd.read_csv('../results/2_higher_dimension/tables/rho_vs_dim.csv')
    for _, row in r2.sort_values(['function', 'dim']).iterrows():
        print(f"  {row['function']} d={row['dim']:<3.0f} rho(A,B)={row['rho_AB']:+.2f}")
except FileNotFoundError:
    print('  exp2 not finished yet')

try:
    r3 = pd.read_csv('../results/3_neural_network/tables/rho_ab.csv')
    print(f"  NN (XOR-MLP) rho(A,B)={r3.iloc[0]['rho_AB']:+.2f}")
except FileNotFoundError:
    print('  exp3 not finished yet')

print()
print('AdamW / SGD+momentum under A vs B')
best1 = pd.read_csv('../results/1_two_dimensional/tables/best_lr.csv')
for opt in ['AdamW', 'SGD_mom']:
    sub = best1[best1['optimizer'] == opt]
    print(f'  {opt}: mean best_A={sub["best_A"].mean():.3f}  mean best_B={sub["best_B"].mean():.3f}')
    for _, row in sub.iterrows():
        rankA = (best1[best1['function'] == row['function']]['best_A'] > row['best_A']).sum() + 1
        rankB = (best1[best1['function'] == row['function']]['best_B'] > row['best_B']).sum() + 1
        print(f"    {row['function']:11s} rank_A={rankA}/6  rank_B={rankB}/6")

print()
print('Criterion C saturation')
for name in ORIGINAL5 + NEW3:
    sub = best1[best1['function'] == name]
    if sub.empty or 'best_C' not in sub.columns:
        print(f'  {name:11s} excluded (no saddle found)')
        continue
    print(f"  {name:11s} best_C range [{sub['best_C'].min():.3f}, {sub['best_C'].max():.3f}]  "
          f"n_at_1.0={(sub['best_C'] > 0.999).sum()}/6")

print()
print('new benchmarks vs original five')
print(pd.read_csv('../results/1_two_dimensional/tables/kendall_w.csv').to_string(index=False))
