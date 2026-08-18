import os, time
import numpy as np
import pandas as pd
import torch
from scipy import stats
import core
import algorithm as see

OUT = '../results/4_medal'
if not os.path.isdir('../results'):
    OUT = 'medal_out'
os.makedirs(OUT, exist_ok=True)

QUICK = os.environ.get('SEE_QUICK') == '1'
if QUICK:
    FUNCS = ['Himmelblau', 'Ackley']
    LRS = [0.2]
    FAMS = ['A', 'B']
    SEED, N, TMAX = 42, 24, 40
else:
    FUNCS = ['Himmelblau', 'Ackley', 'Rastrigin', 'Levy']
    LRS = [0.01, 0.1, 0.2]
    FAMS = ['A', 'B']
    SEED, N, TMAX = 42, 80, 120
IND_OFF = {'A': 1009, 'B': 2017}

t0 = time.time()

def lambda_fn_2d(F):
    def f(Xd):
        H = core.hessian(F, Xd)
        return core.safe_eigh(H, eigenvectors=False).cpu().numpy()[:, 0]
    return f

GEOM = {}
for name in FUNCS:
    F, L = core.FUNCS2D[name]
    ss = core.find_saddles_2d(F, L)
    if not ss:
        print(name, 'no saddle')
        continue
    s = torch.tensor(ss[0]).to(core.DEVICE)
    lmin, lmax, v = core.eigh_min(F, s[None])
    v = v[0] / v[0].norm()
    r_curv = 1 / np.sqrt(abs(lmin.item()))
    f_s = F(s[None])[0].item()
    GEOM[name] = dict(F=F, s=s, v=v, r_curv=r_curv, f_s=f_s, lmin=lmin.item(), lf=lambda_fn_2d(F))
    print(f'{name}: s={s.detach().cpu().numpy().round(3)} lmin={lmin.item():.2f}')
print('saddles', round(time.time() - t0, 1))

shared = {}
mech_rows = []
for name, g in GEOM.items():
    for o in core.OPTS:
        for lr in LRS:
            esc, stp, info = see.simulate(
                g['F'], g['s'], g['v'], g['r_curv'], g['f_s'], o, lr,
                N, TMAX, SEED, FAMS, g['lf'], extra=True)
            shared[(name, o, lr)] = (esc, stp, info)
            hit = ~np.isnan(info['align_at_A'])
            if hit.any():
                al = info['align_at_A'][hit]
                bw = info['b_when_A'][hit]
                mech_rows.append({
                    'function': name, 'optimizer': o, 'lr': lr,
                    'n_A': int(hit.sum()),
                    'mean_align_A': float(al.mean()),
                    'frac_B_already': float(bw.mean()),
                    'mean_align_A_notB': float(al[~bw].mean()) if (~bw).any() else float('nan'),
                    'mean_align_A_andB': float(al[bw].mean()) if bw.any() else float('nan'),
                    'final_mean_lmin': float(np.nanmean(info['final_lmin'])),
                    'final_mean_f': float(np.nanmean(info['final_f'])),
                    'P_A': float(esc[see.headline_idx('A')].mean()),
                    'P_B': float(esc[see.headline_idx('B')].mean()),
                })
    print(name, 'shared', round(time.time() - t0, 1))
pd.DataFrame(mech_rows).to_csv(f'{OUT}/mechanism.csv', index=False)

def best_table(store, seed_tag):
    rows = []
    for name in GEOM:
        for o in core.OPTS:
            rec = {'protocol': seed_tag, 'function': name, 'optimizer': o}
            for fam in FAMS:
                k = see.headline_idx(fam)
                rec[f'best_{fam}'] = max(
                    see.see(store[(name, o, lr)][0][k], store[(name, o, lr)][1][k]) for lr in LRS)
            rows.append(rec)
    return pd.DataFrame(rows)

best_shared = best_table(shared, 'shared')
best_shared.to_csv(f'{OUT}/best_shared.csv', index=False)

indep = {}
for name, g in GEOM.items():
    for o in core.OPTS:
        for lr in LRS:
            for fam in FAMS:
                esc, stp = see.simulate(
                    g['F'], g['s'], g['v'], g['r_curv'], g['f_s'], o, lr,
                    N, TMAX, SEED + IND_OFF[fam], [fam], g['lf'])
                indep[(name, o, lr, fam)] = (esc, stp)
    print(name, 'independent', round(time.time() - t0, 1))

ind_rows = []
for name in GEOM:
    for o in core.OPTS:
        rec = {'protocol': 'independent', 'function': name, 'optimizer': o}
        for fam in FAMS:
            k = see.headline_idx(fam)
            rec[f'best_{fam}'] = max(
                see.see(indep[(name, o, lr, fam)][0][k], indep[(name, o, lr, fam)][1][k]) for lr in LRS)
        ind_rows.append(rec)
best_ind = pd.DataFrame(ind_rows)
best_ind.to_csv(f'{OUT}/best_independent.csv', index=False)

cmp_rows = []
for name in GEOM:
    sub_s = best_shared[best_shared['function'] == name]
    sub_i = best_ind[best_ind['function'] == name]
    a_s, b_s = sub_s['best_A'].values, sub_s['best_B'].values
    a_i, b_i = sub_i['best_A'].values, sub_i['best_B'].values
    rho_s = stats.spearmanr(a_s, b_s).correlation
    rho_i = stats.spearmanr(a_i, b_i).correlation
    uA_s = sub_s.loc[sub_s['best_A'].idxmax(), 'optimizer']
    uB_s = sub_s.loc[sub_s['best_B'].idxmax(), 'optimizer']
    uA_i = sub_i.loc[sub_i['best_A'].idxmax(), 'optimizer']
    uB_i = sub_i.loc[sub_i['best_B'].idxmax(), 'optimizer']
    cmp_rows.append({
        'function': name,
        'rho_AB_shared': rho_s,
        'rho_AB_independent': rho_i,
        'bestA_shared': uA_s, 'bestB_shared': uB_s,
        'bestA_independent': uA_i, 'bestB_independent': uB_i,
        'bestA_flip': uA_s != uA_i,
        'bestB_flip': uB_s != uB_i,
        'pair_disagree_shared': uA_s != uB_s,
        'pair_disagree_independent': uA_i != uB_i,
    })
    print(name, 'rho shared', rho_s, 'ind', rho_i, 'uA', uA_s, uA_i, 'uB', uB_s, uB_i)
pd.DataFrame(cmp_rows).to_csv(f'{OUT}/protocol_compare.csv', index=False)

sel_rows = []
for name, g in GEOM.items():
    sub = best_shared[best_shared['function'] == name]
    uA = sub.loc[sub['best_A'].idxmax(), 'optimizer']
    uB = sub.loc[sub['best_B'].idxmax(), 'optimizer']
    def lr_of(u, fam):
        k = see.headline_idx(fam)
        return max(LRS, key=lambda lr: see.see(shared[(name, u, lr)][0][k], shared[(name, u, lr)][1][k]))
    lrA, lrB = lr_of(uA, 'A'), lr_of(uB, 'B')
    for tag, u, lr in [('pick_A', uA, lrA), ('pick_B', uB, lrB)]:
        esc, stp, info = shared[(name, u, lr)]
        kA, kB = see.headline_idx('A'), see.headline_idx('B')
        sel_rows.append({
            'function': name, 'pick': tag, 'optimizer': u, 'lr': lr,
            'SEE_A': see.see(esc[kA], stp[kA]),
            'SEE_B': see.see(esc[kB], stp[kB]),
            'P_A': float(esc[kA].mean()),
            'P_B': float(esc[kB].mean()),
            'final_mean_lmin': float(np.nanmean(info['final_lmin'])),
            'final_frac_left_saddle': float(np.nanmean(info['final_lmin'] > -1e-3)),
            'final_mean_f': float(np.nanmean(info['final_f'])),
            'final_mean_dist': float(np.nanmean(info['final_dist'])),
        })
    print(name, 'select', uA, lrA, 'vs', uB, lrB)
pd.DataFrame(sel_rows).to_csv(f'{OUT}/selection.csv', index=False)

print()
print('MECHANISM mean align at A-hit, frac already B')
mech = pd.DataFrame(mech_rows)
print(mech.groupby('function')[['mean_align_A', 'frac_B_already']].mean().to_string())
print()
print('PROTOCOL')
print(pd.DataFrame(cmp_rows).to_string(index=False))
print()
print('SELECTION')
print(pd.DataFrame(sel_rows).to_string(index=False))
print('elapsed', round(time.time() - t0, 1))
print('OK')
