"""
N-dimensional trajectory generation + criterion evaluation. Mirrors
see_core.run_config_2d exactly (same VAR list, same per-step logic, same
seed discipline) but uses batched_lanczos_extreme_eigs (Hessian-vector-
product based) for lambda_min instead of the closed-form 2x2 formula, since
that's the only thing that doesn't generalize past n=2. Only lambda_min is
needed per step (criteria B and D); lambda_max and the eigenvector are only
ever needed once, at the saddle itself, during saddle verification/geom
construction -- exactly as in the 2D pipeline.
"""
import torch
import numpy as np


def run_config_nd(F, geom, opt_nm, lr, make_opt, device='cpu', N=200, TMAX=200, seed=42,
                   lanczos_m=30, VAR=None):
    s, v, r_curv, f_s, _ = geom
    n = s.shape[0]
    torch.manual_seed(seed)
    X = (s[None] + 0.1 * torch.randn(N, n, device=device)).requires_grad_(True)
    opt = make_opt(opt_nm, X, lr)
    esc = {k: np.zeros(N, bool) for k in range(len(VAR))}
    stp = {k: np.full(N, TMAX + 1) for k in range(len(VAR))}
    m = min(lanczos_m, n)
    for t in range(TMAX):
        opt.zero_grad(); F(X).sum().backward(); opt.step()
        Xd = X.detach().clone()
        Xd = torch.nan_to_num(Xd, nan=1e6, posinf=1e6, neginf=-1e6)
        lmin, _ = batched_lanczos_extreme_eigs(F, Xd, m=m, device=device)
        lmin = lmin.cpu().numpy()
        fv = F(Xd).detach().cpu().numpy()
        with torch.no_grad():
            dist = (Xd - s).norm(dim=1).cpu().numpy()
            proj = torch.abs((Xd - s) @ v).cpu().numpy()
        for k, (fam, p) in enumerate(VAR):
            if fam == 'A':
                cond = dist > p
            elif fam == 'B':
                cond = lmin > -p
            elif fam == 'C':
                cond = proj > p * r_curv
            else:
                cond = (fv < f_s - p) & (lmin > -1e-3)
            new = (~esc[k]) & cond
            stp[k][new] = t + 1
            esc[k] |= new
    return esc, stp
