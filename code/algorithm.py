import numpy as np
import torch
from scipy import stats
import core

VAR = [
    ('A', 1.5), ('A', 2.0), ('A', 3.0),
    ('B', 1e-2), ('B', 1e-3), ('B', 1e-4),
    ('C', 0.5), ('C', 1.0), ('C', 2.0),
    ('D', 0.25), ('D', 0.5), ('D', 1.0),
]
HEAD = {'A': 2.0, 'B': 1e-3, 'C': 1.0, 'D': 0.5}

def headline_idx(fam):
    return VAR.index((fam, HEAD[fam]))

def oracle(fam, p, dist, lmin, proj, fv, r_curv, f_s):
    if fam == 'A':
        return dist > p
    if fam == 'B':
        return lmin > -p
    if fam == 'C':
        return proj > p * r_curv
    return (fv < f_s - p) & (lmin > -1e-3)

def simulate(F, s, v, r_curv, f_s, opt_name, lr, N, Tmax, seed, families, lambda_fn):
    torch.manual_seed(seed)
    X = (s[None] + 0.1 * torch.randn(N, s.shape[0], device=s.device)).requires_grad_(True)
    opt = core.make_opt(opt_name, X, lr)
    idxs = [k for k, (fam, _) in enumerate(VAR) if fam in families]
    esc = {k: np.zeros(N, bool) for k in idxs}
    stp = {k: np.full(N, Tmax + 1) for k in idxs}
    for t in range(Tmax):
        opt.zero_grad()
        F(X).sum().backward()
        opt.step()
        Xd = X.detach().clone()
        Xd = torch.nan_to_num(Xd, nan=1e4, posinf=1e4, neginf=-1e4)
        lmin = lambda_fn(Xd) if ('B' in families or 'D' in families) else None
        fv = F(Xd).cpu().numpy() if 'D' in families else None
        with torch.no_grad():
            dist = (Xd - s).norm(dim=1).cpu().numpy() if 'A' in families else None
            proj = torch.abs((Xd - s) @ v).cpu().numpy() if 'C' in families else None
        for k in idxs:
            fam, p = VAR[k]
            cond = oracle(fam, p, dist, lmin, proj, fv, r_curv, f_s)
            new = (~esc[k]) & cond
            stp[k][new] = t + 1
            esc[k] |= new
    return esc, stp

run_config = simulate

def see(e, tau):
    return (e.mean() / tau[e].mean()) if e.any() else 0.0

see_pt = see

def see_ci(e, tau, rng, B=2000):
    n = len(e)
    idx = rng.integers(0, n, (B, n))
    E, S = e[idx], tau[idx]
    cnt = E.sum(1)
    mtau = np.where(cnt > 0, np.where(E, S, 0).sum(1) / np.maximum(cnt, 1), np.nan)
    vals = np.where(cnt > 0, E.mean(1) / mtau, 0.0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return see(e, tau), lo, hi

def kendall_w(mat):
    R = np.array([stats.rankdata(row) for row in mat])
    m, n = R.shape
    S = ((R.sum(0) - R.sum() / n) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))

def rank_agreement(mat, names):
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = names.index(b)
            rho = stats.spearmanr(mat[:, i], mat[:, j]).correlation
            pairs.append({'pair': f'{a}-{b}', 'rho': rho})
    return pairs, kendall_w(mat.T)

def evaluate(F, s, v, r_curv, f_s, opt_name, lr, N, Tmax, seed, families, lambda_fn, rng=None):
    esc, stp = simulate(F, s, v, r_curv, f_s, opt_name, lr, N, Tmax, seed, families, lambda_fn)
    out = {}
    for fam in families:
        k = headline_idx(fam)
        rec = {
            'SEE': see(esc[k], stp[k]),
            'P_esc': float(esc[k].mean()),
            'mean_tau': float(stp[k][esc[k]].mean()) if esc[k].any() else float('nan'),
        }
        if rng is not None:
            rec['SEE'], rec['lo'], rec['hi'] = see_ci(esc[k], stp[k], rng)
        out[fam] = rec
    return out, esc, stp

if __name__ == '__main__':
    F, L = core.FUNCS2D['Himmelblau']
    ss = core.find_saddles_2d(F, L)
    s = torch.tensor(ss[0]).to(core.DEVICE)
    lmin, lmax, v = core.eigh_min(F, s[None])
    v = v[0] / v[0].norm()
    r_curv = 1 / np.sqrt(abs(lmin.item()))
    f_s = F(s[None])[0].item()

    def lambda_fn(Xd):
        H = core.hessian(F, Xd)
        return core.safe_eigh(H, eigenvectors=False).cpu().numpy()[:, 0]

    rng = np.random.default_rng(0)
    for name in ['GD', 'Adam']:
        out, _, _ = evaluate(
            F, s, v, r_curv, f_s, name, 0.2,
            N=40, Tmax=50, seed=42, families=['A', 'B'],
            lambda_fn=lambda_fn, rng=rng)
        print(name, {k: round(out[k]['SEE'], 4) for k in out})
