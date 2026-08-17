"""
Sanity check #1 (user-requested): does the Lanczos robustness fix change
escape classification for criteria B/D on trials that DON'T actually
diverge, or does it only affect the pathological rows that would have
crashed anyway? Rerun the exact saved n=2 Styblinski saddle geometry at a
representative LR, once with the fix's current clamp (1e8) and once with an
effectively-permissive clamp (1e15, i.e. clamp essentially never engages for
realistic values), and diff trial-level escape labels for B and D directly.
"""
import sys, pickle, time
sys.path.insert(0, '/Users/arnavdhiman/Projects/SEE-OPT2026-clone/opt2026_ext')
import torch, numpy as np
torch.set_default_dtype(torch.float64)

import see_core
import optimizers as opt_mod
import hessian_eig_nd
import trajectory_nd
# these modules are designed to be flat-bundled into one namespace at kernel-build
# time (see build_kernel.py); replicate that here for local testing.
trajectory_nd.batched_lanczos_extreme_eigs = hessian_eig_nd.batched_lanczos_extreme_eigs

d = pickle.load(open('/Users/arnavdhiman/Projects/SEE-OPT2026-clone/opt2026_ext/kernels/ext2_dimsweep_styblinski/output/results/raw/ext2_Styblinski_n2_raw.pkl', 'rb'))
s_np, v_np, r_curv, f_s, lmin_v = d['geom']
s = torch.tensor(s_np); v = torch.tensor(v_np)
geom = (s, v, r_curv, f_s, lmin_v)
VAR = d['VAR']
OPTS_EXT = d['OPTS_EXT']
SEED, N, TMAX = d['SEED'], d['N'], d['TMAX']

def F_styblinski(X):
    return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(1)

FAMS = ['A', 'B', 'C', 'D']
def headline_idx(fam):
    HEAD = {'A': 2.0, 'B': 1e-3, 'C': 1.0, 'D': 0.5}
    return VAR.index((fam, HEAD[fam]))

import os
LR_TEST = float(os.environ.get('LR_TEST', '0.2'))
device = 'cpu'
print(f'#### TESTING AT LR={LR_TEST} ####')

results = {}
for clamp_label, clamp_val in [('current_fix_1e8', 1e8), ('permissive_1e15', 1e15)]:
    hessian_eig_nd._CLAMP = clamp_val
    # _sanitize's `clamp` parameter defaults to _CLAMP bound AT DEF TIME, so just
    # reassigning the module global above does NOT propagate -- patch the actual
    # bound default so every unqualified _sanitize(t) call picks up the new value.
    hessian_eig_nd._sanitize.__defaults__ = (clamp_val,)
    print(f'\n=== clamp={clamp_label} ({clamp_val:.0e}) ===')
    t0 = time.time()
    per_opt = {}
    for o in OPTS_EXT:
        actual_lr = opt_mod.lr_for(o, LR_TEST)
        e, stp = trajectory_nd.run_config_nd(F_styblinski, geom, o, actual_lr, opt_mod.make_opt,
                                              device=device, N=N, TMAX=TMAX, seed=SEED, lanczos_m=30, VAR=VAR)
        per_opt[o] = (e, stp)
    print(f'  done in {time.time()-t0:.1f}s')
    results[clamp_label] = per_opt

# ---- diff trial-level escape labels for criteria B and D ----
print('\n=== Trial-level diff: current fix (1e8) vs permissive (1e15) ===')
any_diff = False
for fam in ['B', 'D']:
    fi = headline_idx(fam)
    print(f'\n--- criterion {fam} (headline threshold) ---')
    for o in OPTS_EXT:
        e1, s1 = results['current_fix_1e8'][o]
        e2, s2 = results['permissive_1e15'][o]
        esc_diff = (e1[fi] != e2[fi]).sum()
        step_diff_on_agreed = 0
        if esc_diff == 0:
            step_diff_on_agreed = (s1[fi] != s2[fi]).sum()
        n_escaped_1 = e1[fi].sum(); n_escaped_2 = e2[fi].sum()
        flag = '  <-- DIFFERS' if (esc_diff > 0 or step_diff_on_agreed > 0) else ''
        print(f'  {o:13s}: escaped(1e8)={n_escaped_1:3d}/{N}  escaped(1e15)={n_escaped_2:3d}/{N}  '
              f'escape-label mismatches={esc_diff}  step mismatches(where both escaped)={step_diff_on_agreed}{flag}')
        if esc_diff > 0 or step_diff_on_agreed > 0:
            any_diff = True

print(f'\n=== VERDICT: {"CLAMP CHOICE AFFECTS CLASSIFICATION -- investigate further" if any_diff else "IDENTICAL under both clamp settings -- fix only prevents crashes, does not alter escape classification"} ===')
