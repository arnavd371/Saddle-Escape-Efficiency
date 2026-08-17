import sys, pickle, time, os
sys.path.insert(0, '/Users/arnavdhiman/Projects/SEE-OPT2026-clone/opt2026_ext')
import torch, numpy as np
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
SEED, N, TMAX = d['SEED'], d['N'], d['TMAX']

def F_styblinski(X):
    return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(1)

def headline_idx(fam):
    HEAD = {'A': 2.0, 'B': 1e-3, 'C': 1.0, 'D': 0.5}
    return VAR.index((fam, HEAD[fam]))

LR_TEST = float(os.environ.get('LR_TEST', '0.5'))  # test the divergence-prone extreme first
device = 'cpu'
print(f'#### n=50, LR={LR_TEST} ####')

results = {}
for clamp_label, clamp_val in [('current_fix_1e8', 1e8), ('permissive_1e15', 1e15)]:
    hessian_eig_nd._sanitize.__defaults__ = (clamp_val,)
    t0 = time.time()
    per_opt = {}
    for o in OPTS_EXT:
        actual_lr = opt_mod.lr_for(o, LR_TEST)
        e, stp = trajectory_nd.run_config_nd(F_styblinski, geom, o, actual_lr, opt_mod.make_opt,
                                              device=device, N=N, TMAX=TMAX, seed=SEED, lanczos_m=30, VAR=VAR)
        per_opt[o] = (e, stp)
    print(f'  clamp={clamp_label}: done in {time.time()-t0:.1f}s')
    results[clamp_label] = per_opt

any_diff = False
for fam in ['B', 'D']:
    fi = headline_idx(fam)
    print(f'\n--- criterion {fam} ---')
    for o in OPTS_EXT:
        e1, s1 = results['current_fix_1e8'][o]
        e2, s2 = results['permissive_1e15'][o]
        esc_diff = (e1[fi] != e2[fi]).sum()
        step_diff = (s1[fi][e1[fi] & e2[fi]] != s2[fi][e1[fi] & e2[fi]]).sum() if (e1[fi] & e2[fi]).any() else 0
        flag = '  <-- DIFFERS' if (esc_diff > 0 or step_diff > 0) else ''
        print(f'  {o:13s}: escaped(1e8)={e1[fi].sum():3d}/{N}  escaped(1e15)={e2[fi].sum():3d}/{N}  mismatches={esc_diff}  step_mismatches={step_diff}{flag}')
        if esc_diff > 0 or step_diff > 0:
            any_diff = True

print(f'\n=== VERDICT (n=50, LR={LR_TEST}): {"CLAMP AFFECTS CLASSIFICATION" if any_diff else "IDENTICAL -- no effect"} ===')
