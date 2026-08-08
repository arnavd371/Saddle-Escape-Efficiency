"""
N-dimensional saddle finder: random multi-start fsolve, replacing the 2D
grid-search-then-refine approach (a 240^n grid is intractable past n~3-4).
Acceptance thresholds are UNCHANGED from the baseline / see_core.find_saddles_2d:
    ||grad|| < 1e-6, lambda_min(H) < -1e-4, lambda_max(H) > 1e-4
"""
import numpy as np
import torch
from scipy import optimize


def find_saddles_nd(F, L, n, n_starts=5000, keep=3, device='cpu', lanczos_m=30, seed=0):
    rng = np.random.default_rng(seed)

    def gf(p):
        x = torch.tensor(p, device=device)[None].requires_grad_(True)
        F(x).sum().backward()
        return x.grad[0].cpu().numpy()

    starts = rng.uniform(-L, L, size=(n_starts, n))
    found = []
    for p0 in starts:
        try:
            sol, info, ier, _ = optimize.fsolve(gf, p0, full_output=True)
        except Exception:
            continue
        if ier != 1 or np.abs(sol).max() > L or np.linalg.norm(gf(sol)) > 1e-6:
            continue
        # batched_lanczos_extreme_eigs is defined in hessian_eig_nd.py, bundled
        # into the same flat script before this module -- no import needed/possible
        # here since Kaggle script kernels are single-file.
        X = torch.tensor(sol, device=device, dtype=torch.get_default_dtype())[None]
        lmin, lmax = batched_lanczos_extreme_eigs(F, X, m=min(lanczos_m, n), device=device)
        lmin_v, lmax_v = lmin.item(), lmax.item()
        if lmin_v < -1e-4 and lmax_v > 1e-4:
            if all(np.linalg.norm(sol - q) > 0.3 for q in found):
                found.append(sol)
        if len(found) >= keep * 3:  # small over-collection buffer before final sort/keep
            break
    found.sort(key=lambda s: np.linalg.norm(s))
    return [torch.tensor(s, device=device, dtype=torch.get_default_dtype()) for s in found[:keep]]
