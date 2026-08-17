"""
Shared core for the OPT 2026 SEE extension.

This is a device-parametrized extraction of the logic in Cell 1 of
Saddle_EE_Code.ipynb (the original baseline notebook, left untouched).
Function names, criterion definitions (A/B/C/D), acceptance thresholds,
and the bootstrap-CI method are preserved EXACTLY so that new results
are directly comparable to results_final/*.csv from the baseline.

Only additions vs. the notebook:
  - `device` plumbing so this runs on Kaggle GPU as well as CPU.
  - `see_core` is import-only (no top-level execution), so it can be
    reused across the Ext1/Ext2/Ext3 kernel drivers without duplicating
    ~150 lines of criterion-evaluation logic four times.

Nothing here changes the numerical behavior of the original functions.
"""
import torch, numpy as np
from scipy import stats

SEED = 42
N = 200
TMAX = 200
LRS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

# criterion variants (family, parameter) -- identical to the baseline notebook
VAR = [('A', 1.5), ('A', 2.0), ('A', 3.0),        # fixed radius ||x-xs||>r
       ('B', 1e-2), ('B', 1e-3), ('B', 1e-4),     # curvature lmin>-eps
       ('C', 0.5), ('C', 1.0), ('C', 2.0),        # |proj on v_esc| > c*r_curv
       ('D', 0.25), ('D', 0.5), ('D', 1.0)]       # f<f_s-delta AND lmin>-1e-3
HEAD = {'A': 2.0, 'B': 1e-3, 'C': 1.0, 'D': 0.5}   # headline parameter per family
FAMS = ['A', 'B', 'C', 'D']


def headline_idx(fam):
    return VAR.index((fam, HEAD[fam]))


# ---------------------------------------------------------------- 2D benchmark functions
def F_himmelblau(X):
    return (X[:, 0] ** 2 + X[:, 1] - 11) ** 2 + (X[:, 0] + X[:, 1] ** 2 - 7) ** 2


def F_ackley(X):
    a, b, c = 20., 0.2, 2 * np.pi
    return -a * torch.exp(-b * torch.sqrt((X ** 2).mean(1))) - torch.exp(torch.cos(c * X).mean(1)) + a + np.e


def F_rastrigin(X):
    return 20 + (X ** 2 - 10 * torch.cos(2 * np.pi * X)).sum(1)


def F_styblinski(X):
    return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(1)


def F_levy(X):
    W = 1 + (X - 1) / 4
    return (torch.sin(np.pi * W[:, 0]) ** 2 + (W[:, 0] - 1) ** 2 * (1 + 10 * torch.sin(np.pi * W[:, 0] + 1) ** 2)
            + (W[:, 1] - 1) ** 2 * (1 + torch.sin(2 * np.pi * W[:, 1]) ** 2))


FUNCS_2D = {'Himmelblau': F_himmelblau, 'Ackley': F_ackley, 'Rastrigin': F_rastrigin,
            'Styblinski': F_styblinski, 'Levy': F_levy}
DOM_2D = {'Himmelblau': 6., 'Ackley': 5., 'Rastrigin': 5.12, 'Styblinski': 5., 'Levy': 8.}


# ---------------------------------------------------------------- gradient / 2D Hessian eigs
def bgrad(F, X):
    X = X.detach().clone().requires_grad_(True)
    F(X).sum().backward()
    return X.grad.detach()


def beigs(F, X, h=1e-4):
    """Batched 2x2 Hessian eigenvalues by central differences of the gradient.
    X: (n,2). Identical formula to the baseline notebook -- 2D only."""
    dev = X.device
    e1 = torch.tensor([h, 0.], device=dev)
    e2 = torch.tensor([0., h], device=dev)
    gx1 = bgrad(F, X + e1); gx2 = bgrad(F, X - e1)
    gy1 = bgrad(F, X + e2); gy2 = bgrad(F, X - e2)
    Hxx = (gx1[:, 0] - gx2[:, 0]) / (2 * h)
    Hyy = (gy1[:, 1] - gy2[:, 1]) / (2 * h)
    Hxy = ((gx1[:, 1] - gx2[:, 1]) / (2 * h) + (gy1[:, 0] - gy2[:, 0]) / (2 * h)) / 2
    m = (Hxx + Hyy) / 2
    d = torch.sqrt(((Hxx - Hyy) / 2) ** 2 + Hxy ** 2)
    return m - d, m + d, Hxx, Hxy, Hyy


# ---------------------------------------------------------------- saddle finding (2D, grid+fsolve)
def find_saddles_2d(F, L, grid=240, n_cand=800, keep=3, device='cpu'):
    from scipy import optimize
    xs = np.linspace(-L, L, grid)
    G = torch.tensor(np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2), device=device)
    gn = bgrad(F, G).norm(dim=1).cpu().numpy()
    cand = G[np.argsort(gn)[:n_cand]].cpu().numpy()

    def gf(p):
        x = torch.tensor(p, device=device)[None].requires_grad_(True)
        F(x).sum().backward()
        return x.grad[0].cpu().numpy()

    found = []
    for p0 in cand:
        try:
            sol, info, ier, _ = optimize.fsolve(gf, p0, full_output=True)
        except Exception:
            continue
        if ier != 1 or np.abs(sol).max() > L or np.linalg.norm(gf(sol)) > 1e-6:
            continue
        lmin, lmax, *_ = beigs(F, torch.tensor(sol, device=device)[None])
        if lmin.item() < -1e-4 and lmax.item() > 1e-4:
            if all(np.linalg.norm(sol - q) > 0.3 for q in found):
                found.append(sol)
    found.sort(key=lambda s: np.linalg.norm(s))
    return [torch.tensor(s, device=device) for s in found[:keep]]


# ---------------------------------------------------------------- trajectory / criterion evaluation (2D)
def run_config_2d(F, geom, opt_nm, lr, make_opt, device='cpu', N=N, TMAX=TMAX, seed=SEED):
    """Identical semantics to the baseline notebook's run_config, parametrized
    by an injectable make_opt() so new optimizers plug in without touching
    this function."""
    s, v, r_curv, f_s, _ = geom
    torch.manual_seed(seed)
    X = (s[None] + 0.1 * torch.randn(N, 2, device=device)).requires_grad_(True)
    opt = make_opt(opt_nm, X, lr)
    esc = {k: np.zeros(N, bool) for k in range(len(VAR))}
    stp = {k: np.full(N, TMAX + 1) for k in range(len(VAR))}
    for t in range(TMAX):
        opt.zero_grad(); F(X).sum().backward(); opt.step()
        Xd = X.detach().clone()
        Xd = torch.nan_to_num(Xd, nan=1e6, posinf=1e6, neginf=-1e6)
        lmin, _, _, _, _ = beigs(F, Xd); lmin = lmin.cpu().numpy()
        fv = F(Xd).cpu().numpy()
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


# ---------------------------------------------------------------- SEE + stats
def see_pt(e, s):
    return (e.mean() / s[e].mean()) if e.any() else 0.0


def see_ci(e, s, B=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(e)
    idx = rng.integers(0, n, (B, n))
    E = e[idx]; S = s[idx]
    cnt = E.sum(1)
    tau = np.where(cnt > 0, np.where(E, S, 0).sum(1) / np.maximum(cnt, 1), np.nan)
    vals = np.where(cnt > 0, E.mean(1) / tau, 0.0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return see_pt(e, s), (hi - lo) / 2


def kendalls_w(rank_matrix):
    """rank_matrix: (n_judges, n_items) of ranks (as produced by stats.rankdata
    per judge). Identical formula to the baseline notebook."""
    R = np.asarray(rank_matrix)
    m, nit = R.shape
    S = ((R.sum(0) - R.sum() / nit) ** 2).sum()
    return 12 * S / (m ** 2 * (nit ** 3 - nit))


def kendalls_w_ci(value_matrix, B=2000, rng=None, method='bootstrap'):
    """Bootstrap or permutation CI on Kendall's W itself.
    value_matrix: (n_judges, n_items) of raw scores (SEE values), one row
    per criterion/judge, one column per optimizer/item.
    - 'bootstrap': resample items (optimizers) with replacement, B times,
      recompute W each time -> percentile CI. Reported n = n_items.
    - 'permutation': independently permute each judge's ranking B times to
      build a null distribution of W under no agreement, and report the
      p-value of the observed W against that null, plus the null's 95%
      upper interval for reference.
    Returns dict with keys: W, n_items, n_judges, ci_lo, ci_hi (bootstrap)
    or W, n_items, n_judges, p_value, null_97p5 (permutation).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    V = np.asarray(value_matrix, dtype=float)
    m, nit = V.shape
    R_obs = np.array([stats.rankdata(V[j]) for j in range(m)])
    W_obs = kendalls_w(R_obs)

    if method == 'bootstrap':
        Ws = []
        for _ in range(B):
            cols = rng.integers(0, nit, nit)
            Vb = V[:, cols]
            # constant columns after resample can create ties; rankdata handles ties fine
            Rb = np.array([stats.rankdata(Vb[j]) for j in range(m)])
            Ws.append(kendalls_w(Rb))
        Ws = np.array(Ws)
        lo, hi = np.percentile(Ws, [2.5, 97.5])
        return {'W': W_obs, 'n_items': nit, 'n_judges': m, 'ci_lo': lo, 'ci_hi': hi, 'method': 'bootstrap', 'B': B}
    elif method == 'permutation':
        null = []
        for _ in range(B):
            Rp = np.array([rng.permutation(R_obs[j]) for j in range(m)])
            null.append(kendalls_w(Rp))
        null = np.array(null)
        p = (null >= W_obs).mean()
        return {'W': W_obs, 'n_items': nit, 'n_judges': m, 'p_value': p,
                'null_97p5': np.percentile(null, 97.5), 'method': 'permutation', 'B': B}
    else:
        raise ValueError(method)


def spearman_with_n(a, b):
    """Wraps scipy spearmanr but always reports n alongside rho (n is small,
    e.g. 4 in the legacy optimizer set -- carry the caution language)."""
    rho = stats.spearmanr(a, b).correlation
    return rho, len(a)
