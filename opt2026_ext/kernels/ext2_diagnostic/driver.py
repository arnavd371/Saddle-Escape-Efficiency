import time, os, json
import numpy as np, torch

def pick_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        t = torch.zeros(2, 2, device='cuda', dtype=torch.float64)
        (t @ t).sum().item()
        return torch.device('cuda')
    except Exception as ex:
        print(f"[ext2-diag] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext2-diag] device={device}  torch={torch.__version__}  numpy={np.__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
os.makedirs(OUTDIR, exist_ok=True)

N = 200          # trial batch size, matches the real pipeline
TMAX = 200        # steps per trial, matches the real pipeline
LANCZOS_M = 30
FSOLVE_STARTS_FULL = 5000
DIMS_TO_TEST = [5, 10, 25, 50]
FUNCS_TO_TEST = {'Rastrigin': F_rastrigin, 'Ackley': F_ackley, 'Styblinski': F_styblinski}
N_OPT = 7; N_LR = 6; N_FUNC = 3; N_DIMS = 5  # actual Ext2 grid shape

report = {'device': str(device), 'torch': torch.__version__, 'lanczos': {}, 'fsolve': {}, 'accuracy_check': {}}

# ---------------------------------------------------------------- 1. Lanczos lambda_min/lambda_max cost per step
print('\n=== Benchmarking batched Lanczos (lambda_min/lambda_max via HVP) ===')
rng = np.random.default_rng(0)
for n in DIMS_TO_TEST:
    F = F_rastrigin  # representative; row-separable cost is ~function-agnostic (dominated by autodiff graph size ~ n)
    X = torch.tensor(rng.normal(size=(N, n)), device=device)
    # warmup (first call pays CUDA kernel compile / graph trace overhead)
    _ = batched_lanczos_extreme_eigs(F, X, m=LANCZOS_M, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    reps = 3
    t0 = time.time()
    for _ in range(reps):
        lmin, lmax = batched_lanczos_extreme_eigs(F, X, m=LANCZOS_M, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    per_call = (time.time() - t0) / reps
    report['lanczos'][n] = per_call
    print(f'  n={n:3d}: {per_call*1000:.1f} ms/call (batch N={N}, m={LANCZOS_M} Lanczos iters)  '
          f'lmin range=[{lmin.min().item():.2f},{lmin.max().item():.2f}]')

# ---------------------------------------------------------------- 1b. Accuracy check: Lanczos vs dense Hessian (small n, small batch)
print('\n=== Validating Lanczos accuracy vs dense autodiff Hessian (n=5, 10 points) ===')
n_check = 5
Xc = torch.tensor(rng.normal(size=(10, n_check)), device=device)
lmin_lz, lmax_lz = batched_lanczos_extreme_eigs(F_rastrigin, Xc, m=LANCZOS_M, device=device)
lmin_dense, lmax_dense = dense_eigs_reference(F_rastrigin, Xc.cpu())
lmin_err = (lmin_lz.cpu() - lmin_dense).abs().max().item()
lmax_err = (lmax_lz.cpu() - lmax_dense).abs().max().item()
report['accuracy_check'] = {'n': n_check, 'max_abs_err_lmin': lmin_err, 'max_abs_err_lmax': lmax_err}
print(f'  max |lmin_lanczos - lmin_dense| = {lmin_err:.2e}')
print(f'  max |lmax_lanczos - lmax_dense| = {lmax_err:.2e}')
print('  (should be < 1e-6 for m=30 on a well-conditioned n=5 problem; if not, m needs to increase)')

# ---------------------------------------------------------------- 2. fsolve saddle-finding cost (real torch-autodiff gradient, not analytic)
print('\n=== Benchmarking nD fsolve saddle-search (torch-autodiff gradient function) ===')
from scipy import optimize
for n in DIMS_TO_TEST:
    F = F_rastrigin
    L = 5.12

    def gf(p):
        x = torch.tensor(p, device=device)[None].requires_grad_(True)
        F(x).sum().backward()
        return x.grad[0].cpu().numpy()

    n_probe = 30  # small sample, extrapolate to FSOLVE_STARTS_FULL
    starts = rng.uniform(-L, L, size=(n_probe, n))
    t0 = time.time()
    ok = 0
    for p0 in starts:
        try:
            sol, info, ier, _ = optimize.fsolve(gf, p0, full_output=True)
            if ier == 1:
                ok += 1
        except Exception:
            pass
    per_call = (time.time() - t0) / n_probe
    report['fsolve'][n] = per_call
    print(f'  n={n:3d}: {per_call*1000:.1f} ms/call  ({ok}/{n_probe} converged)  '
          f'-> {FSOLVE_STARTS_FULL} starts ~ {per_call*FSOLVE_STARTS_FULL:.1f}s')

# ---------------------------------------------------------------- 3. Extrapolate total Ext2 grid cost
print('\n=== Extrapolated total cost for the full Ext 2 dimensionality sweep ===')
steps_per_config = TMAX
configs_per_dim = N_FUNC * N_OPT * N_LR  # 3*7*6 = 126
total_lanczos_time = 0.0
total_fsolve_time = 0.0
per_dim_report = []
for n in DIMS_TO_TEST + ([2] if 2 not in DIMS_TO_TEST else []):
    lanczos_per_call = report['lanczos'].get(n, report['lanczos'].get(min(DIMS_TO_TEST, key=lambda d: abs(d - n))))
    fsolve_per_call = report['fsolve'].get(n, report['fsolve'].get(min(DIMS_TO_TEST, key=lambda d: abs(d - n))))
    dim_lanczos_time = lanczos_per_call * steps_per_config * configs_per_dim
    dim_fsolve_time = fsolve_per_call * FSOLVE_STARTS_FULL * N_FUNC
    total_lanczos_time += dim_lanczos_time
    total_fsolve_time += dim_fsolve_time
    per_dim_report.append({'n': n, 'lanczos_ms_per_call': lanczos_per_call * 1000,
                            'fsolve_ms_per_call': fsolve_per_call * 1000,
                            'est_trial_grid_seconds': dim_lanczos_time,
                            'est_saddle_search_seconds': dim_fsolve_time})
    print(f'  n={n:3d}: trial-grid ~ {dim_lanczos_time/60:.1f} min   saddle-search ~ {dim_fsolve_time/60:.1f} min')

total_seconds = total_lanczos_time + total_fsolve_time
print(f'\nTOTAL estimated Ext2 runtime (n=2,5,10,25,50; 3 funcs; 7 opts; 6 LRs; N={N}; T={TMAX}; 5000-start fsolve):')
print(f'  saddle-search total: ~{total_fsolve_time/60:.1f} min')
print(f'  trial-grid total:    ~{total_lanczos_time/60:.1f} min')
print(f'  GRAND TOTAL:         ~{total_seconds/60:.1f} min  (~{total_seconds/3600:.2f} hours)')
print(f'  Recommended checkpoint granularity: after each (function, dim) pair '
      f'({len(FUNCS_TO_TEST)*len(DIMS_TO_TEST + [2])} checkpoints total)')

report['extrapolation'] = {'per_dim': per_dim_report, 'total_seconds': total_seconds,
                            'total_hours': total_seconds / 3600}
with open(os.path.join(OUTDIR, 'ext2_diagnostic_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(f'\nDiagnostic report saved to {OUTDIR}/ext2_diagnostic_report.json')
