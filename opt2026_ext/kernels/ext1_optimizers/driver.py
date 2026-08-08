import time, os, pickle
import numpy as np, pandas as pd
import torch
from scipy import stats

def pick_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        t = torch.zeros(2, 2, device='cuda', dtype=torch.float64)
        (t @ t).sum().item()  # a real kernel launch, not just allocation
        return torch.device('cuda')
    except Exception as ex:
        print(f"[ext1] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext1] device = {device}   torch={torch.__version__}  numpy={np.__version__}  scipy={__import__('scipy').__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
RAWDIR = os.path.join(OUTDIR, 'raw')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. saddle verification (identical thresholds to baseline)
print('Locating and verifying saddles (n=7 optimizer extension, functions/thresholds unchanged from baseline)...')
SAD = {}; GEOM = {}
for nm, F in FUNCS_2D.items():
    t0 = time.time()
    ss = find_saddles_2d(F, DOM_2D[nm], device=device)
    SAD[nm] = ss
    if not ss:
        print(f'  {nm}: NONE -> excluded'); continue
    s = ss[0]
    lmin, lmax, Hxx, Hxy, Hyy = beigs(F, s[None])
    lmin_v = lmin.item()
    v = torch.tensor([lmin_v - Hyy.item(), Hxy.item()], device=device) if abs(Hxy.item()) > 1e-12 else torch.tensor([1., 0.], device=device)
    v = v / v.norm()
    r_curv = 1 / np.sqrt(abs(lmin_v))
    f_s = F(s[None])[0].item()
    GEOM[nm] = (s, v, r_curv, f_s, lmin_v)
    print(f'  {nm}: {len(ss)} saddle(s); using {s.cpu().numpy().round(3)}  lambda_min={lmin_v:.2f}  r_curv={r_curv:.3f}  ({time.time()-t0:.1f}s)')
FUNCS_2D_ACTIVE = {k: v for k, v in FUNCS_2D.items() if SAD.get(k)}

# ---------------------------------------------------------------- 2. trajectory generation, all 7 optimizers x 6 LR x 5 functions
print(f'\nRunning all configs: {len(FUNCS_2D_ACTIVE)} functions x {len(OPTS_EXT)} optimizers x {len(LRS)} LRs = '
      f'{len(FUNCS_2D_ACTIVE)*len(OPTS_EXT)*len(LRS)} configs')
DATA = {}
t_start = time.time()
for fn, F in FUNCS_2D_ACTIVE.items():
    for o in OPTS_EXT:
        for base_lr in LRS:
            actual_lr = lr_for(o, base_lr)
            e, s = run_config_2d(F, GEOM[fn], o, actual_lr, make_opt, device=device)
            # key by BASE lr (the shared grid index) so tables line up across optimizers,
            # even though Lion's *actual* lr differs -- actual_lr is stored alongside.
            DATA[(fn, o, base_lr)] = (e, s, actual_lr)
    print(f'  {fn} done. ({time.time()-t_start:.1f}s elapsed)')

# ---------------------------------------------------------------- 2b. Lion robustness check: unscaled LR grid
# Lion's headline results above use LRS*LION_LR_SCALE (paper convention: ~10x
# smaller than Adam). Its SEE came out uniformly low under that convention --
# before reporting "Lion doesn't pattern with RMSProp despite sign-based
# updates" as a finding, check whether that's an LR-scaling artifact by also
# running Lion at the SAME (unscaled) LRS grid as every other optimizer.
print('\nLion robustness check: rerunning at unscaled LRS grid (no 10x scaling)...')
DATA_LION_UNSCALED = {}
t_lion = time.time()
for fn, F in FUNCS_2D_ACTIVE.items():
    for base_lr in LRS:
        e, s = run_config_2d(F, GEOM[fn], 'Lion', base_lr, make_opt, device=device)  # actual_lr == base_lr, unscaled
        DATA_LION_UNSCALED[(fn, base_lr)] = (e, s)
print(f'  Lion unscaled-LR sweep done. ({time.time()-t_lion:.1f}s)')

lion_compare_rows = []
for fn in FUNCS_2D_ACTIVE:
    for fam in FAMS:
        k = headline_idx(fam)
        scaled_best = max(see_pt(DATA[(fn, 'Lion', lr)][0][k], DATA[(fn, 'Lion', lr)][1][k]) for lr in LRS)
        unscaled_best = max(see_pt(DATA_LION_UNSCALED[(fn, lr)][0][k], DATA_LION_UNSCALED[(fn, lr)][1][k]) for lr in LRS)
        lion_compare_rows.append({'function': fn, 'criterion': fam,
                                   'Lion_best_scaled_lr': scaled_best, 'Lion_best_unscaled_lr': unscaled_best,
                                   'diff': unscaled_best - scaled_best})
lion_df = pd.DataFrame(lion_compare_rows)
lion_df.to_csv(os.path.join(OUTDIR, 'lion_lr_robustness_check.csv'), index=False)
print(lion_df.to_string(index=False))
print(f"\nMean Lion SEE, scaled-LR convention:   {lion_df['Lion_best_scaled_lr'].mean():.3f}")
print(f"Mean Lion SEE, unscaled-LR grid:       {lion_df['Lion_best_unscaled_lr'].mean():.3f}")

# save raw trial data (esc/stp arrays) so tables can be regenerated without rerunning
with open(os.path.join(RAWDIR, 'ext1_raw_trials.pkl'), 'wb') as f:
    pickle.dump({'DATA': DATA, 'DATA_LION_UNSCALED': DATA_LION_UNSCALED, 'VAR': VAR,
                 'GEOM': {k: (v[0].cpu().numpy(), v[1].cpu().numpy(), v[2], v[3], v[4]) for k, v in GEOM.items()},
                 'OPTS_EXT': OPTS_EXT, 'LRS': LRS, 'SEED': SEED, 'N': N, 'TMAX': TMAX}, f)
print(f'Raw trial data saved to {RAWDIR}/ext1_raw_trials.pkl')

# ---------------------------------------------------------------- 3. SEE tables (mirrors notebook Cell 1 exactly, OPTS_EXT instead of OPTS)
rows = []
print('\nSEE at lr=0.2  (A_fixed-r / B_curvature / C_eigendisp / D_loss)  [n=7 optimizers]')
for fn in FUNCS_2D_ACTIVE:
    print(f'\n{fn}:')
    for o in OPTS_EXT:
        e, s, actual_lr = DATA[(fn, o, 0.2)]
        vals = []
        for fam in FAMS:
            k = headline_idx(fam)
            v, ci = see_ci(e[k], s[k], rng=rng)
            vals.append((v, ci))
        rows.append({'function': fn, 'optimizer': o, 'lr_grid': 0.2, 'lr_actual': actual_lr,
                      **{f'SEE_{f}': v for f, (v, _) in zip(FAMS, vals)},
                      **{f'CI_{f}': c for f, (_, c) in zip(FAMS, vals)}})
        print(f'  {o:13s}: ' + ' / '.join(f'{v:.3f}±{c:.3f}' for v, c in vals))
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, 'main_lr02_ext.csv'), index=False)

print('\nSEE at each optimizer\'s BEST LR per criterion  [n=7 optimizers]')
best_rows = []
for fn in FUNCS_2D_ACTIVE:
    print(f'\n{fn}:')
    for o in OPTS_EXT:
        vals = []
        for fam in FAMS:
            k = headline_idx(fam)
            v = max(see_pt(DATA[(fn, o, lr)][0][k], DATA[(fn, o, lr)][1][k]) for lr in LRS)
            vals.append(v)
        best_rows.append({'function': fn, 'optimizer': o, **{f'best_{f}': v for f, v in zip(FAMS, vals)}})
        print(f'  {o:13s}: ' + ' / '.join(f'{v:.3f}' for v in vals))
pd.DataFrame(best_rows).to_csv(os.path.join(OUTDIR, 'best_lr_ext.csv'), index=False)

# ---------------------------------------------------------------- 4. cross-criterion stats (n now 6-7 instead of 4 -- report n explicitly)
print(f'\n pairwise Spearman between criteria (best-LR ranks), n={len(OPTS_EXT)} optimizers ')
spearman_rows = []
for fn in FUNCS_2D_ACTIVE:
    sub = [r for r in best_rows if r['function'] == fn]
    a = {f: [r['best_' + f] for r in sub] for f in FAMS}
    for i, f1 in enumerate(FAMS):
        for j, f2 in enumerate(FAMS):
            if j <= i:
                continue
            rho, n_used = spearman_with_n(a[f1], a[f2])
            spearman_rows.append({'function': fn, 'pair': f'{f1}-{f2}', 'rho': rho, 'n': n_used})
    print(f"  {fn:11s} A-B:{stats.spearmanr(a['A'],a['B']).correlation:+.2f} "
          f"A-C:{stats.spearmanr(a['A'],a['C']).correlation:+.2f} "
          f"A-D:{stats.spearmanr(a['A'],a['D']).correlation:+.2f} "
          f"B-D:{stats.spearmanr(a['B'],a['D']).correlation:+.2f}   (n={len(OPTS_EXT)})")
pd.DataFrame(spearman_rows).to_csv(os.path.join(OUTDIR, 'spearman_pairwise_ext.csv'), index=False)

print(f"\nKendall's W concordance across the 4 criteria (best-LR), n={len(OPTS_EXT)} optimizers, with bootstrap CI")
w_rows = []
for fn in FUNCS_2D_ACTIVE:
    sub = [r for r in best_rows if r['function'] == fn]
    V = np.array([[r['best_' + f] for r in sub] for f in FAMS])  # (4 judges, n_opt items)
    res = kendalls_w_ci(V, rng=rng, method='bootstrap')
    w_rows.append({'function': fn, **res})
    print(f"  {fn:11s} W={res['W']:.2f}  95% CI [{res['ci_lo']:.2f}, {res['ci_hi']:.2f}]  (n_items={res['n_items']})")
pd.DataFrame(w_rows).to_csv(os.path.join(OUTDIR, 'kendall_w_ext.csv'), index=False)

print('\nwithin-family threshold stability (Spearman across variants, best-LR)')
stability_rows = []
for fam, ps in [('A', [1.5, 2.0, 3.0]), ('B', [1e-2, 1e-3, 1e-4]), ('C', [0.5, 1.0, 2.0]), ('D', [0.25, 0.5, 1.0])]:
    rhos = []
    for fn in FUNCS_2D_ACTIVE:
        rank = {}
        for p in ps:
            k = VAR.index((fam, p))
            rank[p] = [max(see_pt(DATA[(fn, o, lr)][0][k], DATA[(fn, o, lr)][1][k]) for lr in LRS) for o in OPTS_EXT]
        r1, n_used = spearman_with_n(rank[ps[0]], rank[ps[-1]])
        if not np.isnan(r1):
            rhos.append(r1)
            stability_rows.append({'family': fam, 'function': fn, 'rho': r1, 'n': n_used})
    print(f'  family {fam}: mean rho(extreme thresholds) = {np.mean(rhos):+.2f} over {len(rhos)} functions  (n={len(OPTS_EXT)} each)')
pd.DataFrame(stability_rows).to_csv(os.path.join(OUTDIR, 'within_family_stability_ext.csv'), index=False)

# ---------------------------------------------------------------- 5. AdamW-vs-Adam / Lion-vs-RMSProp qualitative pattern check
print('\nQualitative pattern check: does AdamW behave like Adam? Does Lion pattern with RMSProp?')
pattern_rows = []
for fn in FUNCS_2D_ACTIVE:
    sub = {r['optimizer']: r for r in best_rows if r['function'] == fn}
    for fam in FAMS:
        adam_v, adamw_v = sub['Adam']['best_' + fam], sub['AdamW']['best_' + fam]
        rms_v, lion_v = sub['RMSProp']['best_' + fam], sub['Lion']['best_' + fam]
        pattern_rows.append({'function': fn, 'criterion': fam,
                              'Adam': adam_v, 'AdamW': adamw_v, 'abs_diff_Adam_AdamW': abs(adam_v - adamw_v),
                              'RMSProp': rms_v, 'Lion': lion_v, 'abs_diff_RMSProp_Lion': abs(rms_v - lion_v)})
pd.DataFrame(pattern_rows).to_csv(os.path.join(OUTDIR, 'adamw_lion_pattern_check_ext.csv'), index=False)
print(pd.DataFrame(pattern_rows).groupby('criterion')[['abs_diff_Adam_AdamW', 'abs_diff_RMSProp_Lion']].mean().to_string())

print(f'\nExt1 done. Total elapsed: {time.time()-t_start:.1f}s. Files in {OUTDIR}/')
