import sys, pickle, time
sys.path.insert(0, '/Users/arnavdhiman/Projects/SEE-OPT2026-clone/opt2026_ext')
import torch, numpy as np
from scipy import stats
torch.set_default_dtype(torch.float64)

import hessian_eig_nd
import trajectory_nd
import optimizers as opt_mod
trajectory_nd.batched_lanczos_extreme_eigs = hessian_eig_nd.batched_lanczos_extreme_eigs

d = pickle.load(open('/Users/arnavdhiman/Projects/SEE-OPT2026-clone/opt2026_ext/kernels/ext2_dimsweep_styblinski/output/results/raw/ext2_Styblinski_n50_raw.pkl', 'rb'))
s_np, v_np, r_curv, f_s, lmin_v = d['geom']
s = torch.tensor(s_np); v = torch.tensor(v_np)
geom = (s, v, r_curv, f_s, lmin_v)
VAR = d['VAR']
OPTS_EXT = d['OPTS_EXT']
LRS = d['LRS']
SEED, N, TMAX = d['SEED'], d['N'], d['TMAX']

def F_styblinski(X):
    return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(1)

def headline_idx(fam):
    HEAD = {'A': 2.0, 'B': 1e-3, 'C': 1.0, 'D': 0.5}
    return VAR.index((fam, HEAD[fam]))

def see_pt(e, s):
    return (e.mean() / s[e].mean()) if e.any() else 0.0

device = 'cpu'
FAMS = ['A', 'B', 'C', 'D']

for clamp_label, clamp_val in [('current_fix_1e8', 1e8), ('permissive_1e15', 1e15)]:
    hessian_eig_nd._sanitize.__defaults__ = (clamp_val,)
    t0 = time.time()
    DATA = {}
    for o in OPTS_EXT:
        for base_lr in LRS:
            actual_lr = opt_mod.lr_for(o, base_lr)
            e, stp = trajectory_nd.run_config_nd(F_styblinski, geom, o, actual_lr, opt_mod.make_opt,
                                                  device=device, N=N, TMAX=TMAX, seed=SEED, lanczos_m=30, VAR=VAR)
            DATA[(o, base_lr)] = (e, stp)
        print(f'  [{clamp_label}] {o} done ({time.time()-t0:.1f}s elapsed)', flush=True)

    best = {}
    for o in OPTS_EXT:
        vals = []
        for fam in FAMS:
            fi = headline_idx(fam)
            v_best = max(see_pt(DATA[(o, lr)][0][fi], DATA[(o, lr)][1][fi]) for lr in LRS)
            vals.append(v_best)
        best[o] = vals

    print(f'\n=== {clamp_label}: best-LR SEE per optimizer (A,B,C,D) ===')
    for o in OPTS_EXT:
        print(f'  {o:13s}: ' + ' / '.join(f'{v:.3f}' for v in best[o]))

    a_b = [best[o][1] for o in OPTS_EXT]
    a_d = [best[o][3] for o in OPTS_EXT]
    rho_bd = stats.spearmanr(a_b, a_d).correlation
    a_a = [best[o][0] for o in OPTS_EXT]
    rho_ab = stats.spearmanr(a_a, a_b).correlation
    print(f'  A-B rho = {rho_ab:+.3f}   B-D rho = {rho_bd:+.3f}   (total {time.time()-t0:.1f}s)')
    print()
