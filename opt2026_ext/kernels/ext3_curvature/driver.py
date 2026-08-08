import time, os, pickle
import numpy as np, pandas as pd
import torch
from scipy import stats

def pick_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        t = torch.zeros(2, 2, device='cuda', dtype=torch.float64)
        (t @ t).sum().item()
        return torch.device('cuda')
    except Exception as ex:
        print(f"[ext3] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext3] device={device}  torch={torch.__version__}  numpy={np.__version__}  scipy={__import__('scipy').__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
RAWDIR = os.path.join(OUTDIR, 'raw')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)
rng = np.random.default_rng(SEED)

print(f'\nContinuous curvature sweep: {len(K_VALUES)} k-values from {K_VALUES[0]} to {K_VALUES[-1]}, '
      f'{len(OPTS_EXT)} optimizers x {len(LRS)} LRs = {len(K_VALUES)*len(OPTS_EXT)*len(LRS)} configs')

# ---------------------------------------------------------------- 1. verify saddle at origin for every k
print('\nVerifying saddle at (0,0) for every k (analytic + finite-difference cross-check)...')
verify_rows = []
GEOM_K = {}
for k in K_VALUES:
    ok, geom, diag = verify_and_build_geom(k, device=device)
    GEOM_K[k] = geom
    verify_rows.append({'k': k, 'accepted': ok, **diag})
    if not ok:
        print(f'  k={k:.3f}: FAILED acceptance thresholds -- {diag}')
pd.DataFrame(verify_rows).to_csv(os.path.join(OUTDIR, 'curvature_saddle_verification.csv'), index=False)
n_failed = sum(1 for r in verify_rows if not r['accepted'])
print(f'  {len(K_VALUES)-n_failed}/{len(K_VALUES)} k-values verified as saddles at (0,0). '
      f'(analytic-vs-finite-diff lmin agreement: max diff = {max(r["analytic_vs_fd_lmin_diff"] for r in verify_rows):.2e})')
K_VALUES_OK = [r['k'] for r in verify_rows if r['accepted']]

# ---------------------------------------------------------------- 2. trajectory generation across k x optimizer x LR
DATA = {}
t_start = time.time()
for k in K_VALUES_OK:
    F = make_F_curv(k)
    geom = GEOM_K[k]
    for o in OPTS_EXT:
        for base_lr in LRS:
            actual_lr = lr_for(o, base_lr)
            e, s = run_config_2d(F, geom, o, actual_lr, make_opt, device=device)
            DATA[(k, o, base_lr)] = (e, s, actual_lr)
    print(f'  k={k:.3f} done. ({time.time()-t_start:.1f}s elapsed)')

with open(os.path.join(RAWDIR, 'ext3_raw_trials.pkl'), 'wb') as f:
    pickle.dump({'DATA': DATA, 'VAR': VAR, 'K_VALUES': K_VALUES_OK,
                 'OPTS_EXT': OPTS_EXT, 'LRS': LRS, 'SEED': SEED, 'N': N, 'TMAX': TMAX,
                 'A_COEF': A_COEF, 'B_COEF': B_COEF, 'OMEGA': OMEGA}, f)
print(f'Raw trial data saved to {RAWDIR}/ext3_raw_trials.pkl')

# ---------------------------------------------------------------- 3. best-LR SEE per (k, optimizer, criterion)
best_rows = []
for k in K_VALUES_OK:
    for o in OPTS_EXT:
        vals = []
        for fam in FAMS:
            fi = headline_idx(fam)
            v = max(see_pt(DATA[(k, o, lr)][0][fi], DATA[(k, o, lr)][1][fi]) for lr in LRS)
            vals.append(v)
        best_rows.append({'k': k, 'optimizer': o, **{f'best_{f}': v for f, v in zip(FAMS, vals)}})
pd.DataFrame(best_rows).to_csv(os.path.join(OUTDIR, 'curvature_best_lr.csv'), index=False)

# ---------------------------------------------------------------- 4. Kendall's W (with bootstrap CI) vs k -- the headline curve
print("\nKendall's W across the 4 criteria, per k, with bootstrap CI")
w_rows = []
for k in K_VALUES_OK:
    sub = [r for r in best_rows if r['k'] == k]
    V = np.array([[r['best_' + f] for r in sub] for f in FAMS])  # (4 judges, n_opt items)
    res = kendalls_w_ci(V, rng=rng, method='bootstrap')
    w_rows.append({'k': k, **res})
    print(f'  k={k:6.3f}  W={res["W"]:.3f}  95% CI [{res["ci_lo"]:.3f}, {res["ci_hi"]:.3f}]  (n_items={res["n_items"]})')
w_df = pd.DataFrame(w_rows)
w_df.to_csv(os.path.join(OUTDIR, 'curvature_kendall_w_vs_k.csv'), index=False)

# ---------------------------------------------------------------- 5. correlation between k and W, with bootstrap CI
print('\nCorrelation between oscillation-density parameter k and Kendall\'s W')
ks = w_df['k'].values
Ws = w_df['W'].values
rho_obs, n_used = spearman_with_n(ks, Ws)
B = 2000
boot_rhos = []
idx_all = np.arange(len(ks))
for _ in range(B):
    idx = rng.integers(0, len(ks), len(ks))
    r = stats.spearmanr(ks[idx], Ws[idx]).correlation
    if not np.isnan(r):
        boot_rhos.append(r)
boot_rhos = np.array(boot_rhos)
ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
print(f'  Spearman rho(k, W) = {rho_obs:+.3f}  (n={n_used} k-values)  95% bootstrap CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')
with open(os.path.join(OUTDIR, 'curvature_k_vs_W_correlation.csv'), 'w') as f:
    f.write('rho,n,ci_lo,ci_hi,B\n')
    f.write(f'{rho_obs},{n_used},{ci_lo},{ci_hi},{B}\n')

print(f'\nExt3 done. Total elapsed: {time.time()-t_start:.1f}s. Files in {OUTDIR}/')
