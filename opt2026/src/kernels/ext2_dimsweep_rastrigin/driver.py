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
        print(f"[ext2] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext2] device={device}  torch={torch.__version__}  numpy={np.__version__}  scipy={__import__('scipy').__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
RAWDIR = os.path.join(OUTDIR, 'raw')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)
rng = np.random.default_rng(SEED)

FUNC_NAME = "Rastrigin"   # substituted per kernel (Ackley / Rastrigin / Styblinski)
F = FUNCS_ND[FUNC_NAME]
L = DOM_ND[FUNC_NAME]
LANCZOS_M = 30
N_STARTS = 5000

print(f'=== Ext2 dimensionality sweep: {FUNC_NAME} across n={DIMS} ===')
print(f'(checkpointing after every dimension -- partial progress survives a crash or timeout on a later dim)')

all_best_rows = []
all_spearman_rows = []
all_w_rows = []
saddle_report_rows = []
t_start = time.time()

for n in DIMS:
    t_dim = time.time()
    print(f'\n--- {FUNC_NAME} n={n} ---')
    saddles = find_saddles_nd(F, L, n, n_starts=N_STARTS, keep=3, device=device, seed=SEED)
    if not saddles:
        print(f'  NO SADDLE FOUND at n={n} ({N_STARTS} random starts) -- skipping this dimension')
        saddle_report_rows.append({'function': FUNC_NAME, 'n': n, 'n_saddles_found': 0})
        pd.DataFrame(saddle_report_rows).to_csv(os.path.join(OUTDIR, f'ext2_{FUNC_NAME}_saddle_report.csv'), index=False)
        continue

    s = saddles[0]
    X_s = s[None]
    lmin_t, lmax_t, vmin_t = batched_lanczos_min_eigpair(F, X_s, m=min(LANCZOS_M, n), device=device)
    lmin_v, lmax_v = lmin_t.item(), lmax_t.item()
    r_curv = 1 / np.sqrt(abs(lmin_v))
    f_s = F(X_s)[0].item()
    geom = (s, vmin_t[0], r_curv, f_s, lmin_v)
    saddle_report_rows.append({'function': FUNC_NAME, 'n': n, 'n_saddles_found': len(saddles),
                                'lambda_min': lmin_v, 'lambda_max': lmax_v, 'r_curv': r_curv})
    pd.DataFrame(saddle_report_rows).to_csv(os.path.join(OUTDIR, f'ext2_{FUNC_NAME}_saddle_report.csv'), index=False)
    print(f'  saddle verified: {len(saddles)} candidate(s); lambda_min={lmin_v:.3f} lambda_max={lmax_v:.3f} '
          f'r_curv={r_curv:.4f}  ({time.time()-t_dim:.1f}s)')

    DATA_n = {}
    for o in OPTS_EXT:
        for base_lr in LRS:
            actual_lr = lr_for(o, base_lr)
            e, stp = run_config_nd(F, geom, o, actual_lr, make_opt, device=device, N=N, TMAX=TMAX,
                                    seed=SEED, lanczos_m=LANCZOS_M, VAR=VAR)
            DATA_n[(o, base_lr)] = (e, stp, actual_lr)
    print(f'  trial grid done ({len(OPTS_EXT)}x{len(LRS)}={len(OPTS_EXT)*len(LRS)} configs, {time.time()-t_dim:.1f}s total for this dim)')

    # ---- checkpoint: raw trial data + accumulated summary CSVs, written NOW (not at the end) ----
    with open(os.path.join(RAWDIR, f'ext2_{FUNC_NAME}_n{n}_raw.pkl'), 'wb') as f:
        pickle.dump({'DATA': DATA_n, 'geom': (s.cpu().numpy(), vmin_t[0].cpu().numpy(), r_curv, f_s, lmin_v),
                     'n': n, 'FUNC_NAME': FUNC_NAME, 'OPTS_EXT': OPTS_EXT, 'LRS': LRS, 'VAR': VAR,
                     'SEED': SEED, 'N': N, 'TMAX': TMAX}, f)

    best_rows_n = []
    for o in OPTS_EXT:
        vals = []
        for fam in FAMS:
            fi = headline_idx(fam)
            v_best = max(see_pt(DATA_n[(o, lr)][0][fi], DATA_n[(o, lr)][1][fi]) for lr in LRS)
            vals.append(v_best)
        best_rows_n.append({'function': FUNC_NAME, 'n': n, 'optimizer': o, **{f'best_{f}': v for f, v in zip(FAMS, vals)}})
    all_best_rows += best_rows_n
    pd.DataFrame(all_best_rows).to_csv(os.path.join(OUTDIR, f'ext2_{FUNC_NAME}_best_lr.csv'), index=False)

    a = {f: [r['best_' + f] for r in best_rows_n] for f in FAMS}
    rho_ab, n_ab = spearman_with_n(a['A'], a['B'])
    rho_bd, n_bd = spearman_with_n(a['B'], a['D'])
    all_spearman_rows.append({'function': FUNC_NAME, 'n': n, 'rho_AB': rho_ab, 'rho_BD': rho_bd,
                               'abs_rho_AB': abs(rho_ab) if rho_ab == rho_ab else np.nan,
                               'one_minus_rho_BD': 1 - rho_bd if rho_bd == rho_bd else np.nan,
                               'n_optimizers': n_ab})
    pd.DataFrame(all_spearman_rows).to_csv(os.path.join(OUTDIR, f'ext2_{FUNC_NAME}_spearman_vs_dim.csv'), index=False)
    print(f'  n={n}: A-B rho={rho_ab:+.3f}   B-D rho={rho_bd:+.3f}   (n_optimizers={n_ab})')

    V = np.array([[r['best_' + f] for r in best_rows_n] for f in FAMS])
    w_res = kendalls_w_ci(V, rng=rng, method='bootstrap')
    all_w_rows.append({'function': FUNC_NAME, 'n': n, **w_res})
    pd.DataFrame(all_w_rows).to_csv(os.path.join(OUTDIR, f'ext2_{FUNC_NAME}_kendall_w_vs_dim.csv'), index=False)
    print(f'  n={n}: Kendall W={w_res["W"]:.3f}  95% CI [{w_res["ci_lo"]:.3f},{w_res["ci_hi"]:.3f}]')

    print(f'  --- n={n} checkpoint written. total elapsed {(time.time()-t_start)/60:.1f} min ---')

print(f'\n{FUNC_NAME} dimensionality sweep complete. Total: {(time.time()-t_start)/60:.1f} min')
print(f'Files in {OUTDIR}/ (ext2_{FUNC_NAME}_*.csv) and {RAWDIR}/ (per-dim raw pickles)')
