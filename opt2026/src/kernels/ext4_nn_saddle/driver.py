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
        print(f"[ext4] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext4] device={device}  torch={torch.__version__}  numpy={np.__version__}  scipy={__import__('scipy').__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
RAWDIR = os.path.join(OUTDIR, 'raw')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)
rng = np.random.default_rng(SEED)

print(f'\n=== Ext4: first real (non-benchmark) loss landscape -- XOR MLP, {N_PARAMS}-dim parameter space ===')
print('Saddle constructed via tied-unit symmetry (see xor_network.py docstring for why global')
print('multi-start fsolve fails on this saturating landscape).')

# ---------------------------------------------------------------- 1. saddle: CPU-verified fixed coordinates
# The saddle construction (find_xor_saddle) is a fragile 4000-step iterative
# tied-subspace gradient descent -- of 6 random seeds tried locally, only
# seed=2 converges to the clean saddle (loss=ln(2), lambda_min=-0.0152); the
# other 5 land in a degenerate attractor (lambda_min=lambda_max~0, huge
# param magnitude). A first Kaggle GPU run (kernel version 1) reproduced
# that SAME degenerate failure even with seed=2 -- GPU/CPU float64
# non-determinism compounded over 4000 steps landed in a different basin
# than the CPU run did (this is exactly the class of non-determinism the
# baseline README already documents as a known limitation). Rather than gamble
# on construction being GPU-reproducible, the fragile part (construction) is
# done once on CPU and hardcoded here; only the trial-grid forward simulation
# (which every other kernel in this project already does successfully on
# GPU from a fixed starting point) runs on GPU.
_THETA_SADDLE = [-1.0894063429134349e-15, -1.0894063429134349e-15, -1.0894063429134349e-15,
                 -1.0894063429134349e-15, -0.11738607773232793, -0.11738607773232793,
                 -0.5271470428799861, -0.5271470428799861, -0.12319411588445499]
s = torch.tensor(_THETA_SADDLE, device=device)
lmin_t, lmax_t, vmin_t = batched_lanczos_min_eigpair(F_xor_loss, s[None], m=N_PARAMS, device=device)
lmin_v, lmax_v = lmin_t.item(), lmax_t.item()
r_curv = 1 / np.sqrt(abs(lmin_v))
f_s = F_xor_loss(s[None])[0].item()
geom = (s, vmin_t[0], r_curv, f_s, lmin_v)

is_saddle = lmin_v < -1e-4 and lmax_v > 1e-4
print(f'\nFixed (CPU-verified) saddle recomputed on {device}: loss={f_s:.6f}  lambda_min={lmin_v:.6f}  '
      f'lambda_max={lmax_v:.6f}  r_curv={r_curv:.4f}')
print(f'  ACCEPTED AS SADDLE (lmin<-1e-4 and lmax>1e-4): {is_saddle}')
print(f'  (CPU reference values: loss=0.693147 lambda_min=-0.015189 lambda_max=0.462421 -- '
      f'comparing to catch any residual device-dependent eigenvalue drift)')
if not is_saddle:
    raise RuntimeError('Fixed saddle point failed acceptance on this device -- do not proceed with trial grid.')

pd.DataFrame([{'loss': f_s, 'lambda_min': lmin_v, 'lambda_max': lmax_v, 'r_curv': r_curv,
               'n_params': N_PARAMS, 'device': str(device)}]
             ).to_csv(os.path.join(OUTDIR, 'ext4_xor_saddle_report.csv'), index=False)

# ---------------------------------------------------------------- 2. trial grid: all 7 optimizers x 6 LRs x 4 criteria
print(f'\nRunning trial grid: {len(OPTS_EXT)} optimizers x {len(LRS)} LRs = {len(OPTS_EXT)*len(LRS)} configs, '
      f'N={N} trials, T={TMAX} steps')
DATA = {}
t_start = time.time()
for o in OPTS_EXT:
    for base_lr in LRS:
        actual_lr = lr_for(o, base_lr)
        e, stp = run_config_nd(F_xor_loss, geom, o, actual_lr, make_opt, device=device, N=N, TMAX=TMAX,
                                seed=SEED, lanczos_m=N_PARAMS, VAR=VAR)
        DATA[(o, base_lr)] = (e, stp, actual_lr)
    print(f'  {o} done. ({time.time()-t_start:.1f}s elapsed)')

with open(os.path.join(RAWDIR, 'ext4_xor_raw_trials.pkl'), 'wb') as f:
    pickle.dump({'DATA': DATA, 'geom': (s.cpu().numpy(), vmin_t[0].cpu().numpy(), r_curv, f_s, lmin_v),
                 'VAR': VAR, 'OPTS_EXT': OPTS_EXT, 'LRS': LRS, 'SEED': SEED, 'N': N, 'TMAX': TMAX,
                 'N_PARAMS': N_PARAMS, 'theta_saddle': _THETA_SADDLE}, f)
print(f'Raw trial data saved to {RAWDIR}/ext4_xor_raw_trials.pkl')

# ---------------------------------------------------------------- 3. SEE tables
best_rows = []
for o in OPTS_EXT:
    vals = []
    for fam in FAMS:
        fi = headline_idx(fam)
        v_best = max(see_pt(DATA[(o, lr)][0][fi], DATA[(o, lr)][1][fi]) for lr in LRS)
        vals.append(v_best)
    best_rows.append({'optimizer': o, **{f'best_{f}': v for f, v in zip(FAMS, vals)}})
    print(f'  {o:13s}: ' + ' / '.join(f'{v:.3f}' for v in vals))
df_best = pd.DataFrame(best_rows)
df_best.to_csv(os.path.join(OUTDIR, 'ext4_xor_best_lr.csv'), index=False)

# ---------------------------------------------------------------- 4. the key question: does B-D still agree, does A-B still disagree?
print('\n=== The key question: does the criterion-dependence pattern replicate on a real network? ===')
spearman_rows = []
for f1, f2 in [('A', 'B'), ('A', 'D'), ('B', 'D'), ('A', 'C'), ('B', 'C'), ('C', 'D')]:
    a = df_best[f'best_{f1}'].values
    b = df_best[f'best_{f2}'].values
    rho, n_used = spearman_with_n(a, b)
    spearman_rows.append({'pair': f'{f1}-{f2}', 'rho': rho, 'n': n_used})
    print(f'  {f1}-{f2}: rho={rho:+.3f}  (n={n_used})')
pd.DataFrame(spearman_rows).to_csv(os.path.join(OUTDIR, 'ext4_xor_spearman.csv'), index=False)

V = np.array([[r[f'best_{f}'] for r in best_rows] for f in FAMS])
w_res = kendalls_w_ci(V, rng=rng, method='bootstrap')
print(f"\n  Kendall's W = {w_res['W']:.3f}  95% CI [{w_res['ci_lo']:.3f}, {w_res['ci_hi']:.3f}]  (n_items={w_res['n_items']})")
pd.DataFrame([{'W': w_res['W'], 'n_items': w_res['n_items'], 'n_judges': w_res['n_judges'],
               'ci_lo': w_res['ci_lo'], 'ci_hi': w_res['ci_hi'], 'method': w_res['method'], 'B': w_res['B']}]
             ).to_csv(os.path.join(OUTDIR, 'ext4_xor_kendall_w.csv'), index=False)

print(f'\nExt4 done. Total elapsed: {time.time()-t_start:.1f}s. Files in {OUTDIR}/')
