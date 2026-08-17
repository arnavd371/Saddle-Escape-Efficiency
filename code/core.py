import numpy as np
import torch
from scipy import optimize, stats

torch.set_default_dtype(torch.float64)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    try:
        x = torch.ones(2, 2, device=DEVICE, requires_grad=True)
        (x * x).sum().backward()
        _ = x.grad.cpu().numpy()
    except Exception as e:
        print(f'cuda unusable ({e}), falling back to cpu')
        DEVICE = torch.device('cpu')
print('device:', DEVICE)

def f_himmelblau(X):
    return (X[:, 0] ** 2 + X[:, 1] - 11) ** 2 + (X[:, 0] + X[:, 1] ** 2 - 7) ** 2

def f_ackley(X):
    a, b, c = 20., 0.2, 2 * np.pi
    return (-a * torch.exp(-b * torch.sqrt((X ** 2).mean(1)))
            - torch.exp(torch.cos(c * X).mean(1)) + a + np.e)

def f_rastrigin(X):
    return 10 * X.shape[1] + (X ** 2 - 10 * torch.cos(2 * np.pi * X)).sum(1)

def f_styblinski(X):
    return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(1)

def f_levy(X):
    W = 1 + (X - 1) / 4
    return (torch.sin(np.pi * W[:, 0]) ** 2
            + (W[:, 0] - 1) ** 2 * (1 + 10 * torch.sin(np.pi * W[:, 0] + 1) ** 2)
            + (W[:, 1] - 1) ** 2 * (1 + torch.sin(2 * np.pi * W[:, 1]) ** 2))

def f_beale(X):
    x, y = X[:, 0], X[:, 1]
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

def f_booth(X):
    x, y = X[:, 0], X[:, 1]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

def f_schwefel(X):
    d = X.shape[1]
    return 418.9829 * d - (X * torch.sin(torch.sqrt(torch.abs(X)))).sum(1)

FUNCS2D = {
    'Himmelblau': (f_himmelblau, 6.0),
    'Ackley': (f_ackley, 5.0),
    'Rastrigin': (f_rastrigin, 5.12),
    'Styblinski': (f_styblinski, 5.0),
    'Levy': (f_levy, 8.0),
    'Beale': (f_beale, 4.5),
    'Booth': (f_booth, 10.0),
    'Schwefel': (f_schwefel, 500.0),
}

def grad(F, X):
    X = X.detach().clone().requires_grad_(True)
    F(X).sum().backward()
    return X.grad.detach()

def hvp(F, X, v):
    X = X.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(F(X).sum(), X, create_graph=True)[0]
    return torch.autograd.grad((g * v).sum(), X)[0].detach()

def hessian(F, X):
    n, d = X.shape
    cols = []
    for i in range(d):
        e = torch.zeros_like(X)
        e[:, i] = 1.
        cols.append(hvp(F, X, e))
    return torch.stack(cols, -1)

def safe_eigh(H, eigenvectors=True):
    fn = torch.linalg.eigh if eigenvectors else torch.linalg.eigvalsh
    try:
        return fn(H)
    except Exception:
        n, d = H.shape[0], H.shape[-1]
        evals = torch.full((n, d), float('nan'), device=H.device)
        evecs = torch.full((n, d, d), float('nan'), device=H.device) if eigenvectors else None
        for i in range(n):
            try:
                r = fn(H[i:i + 1])
                if eigenvectors:
                    evals[i], evecs[i] = r[0][0], r[1][0]
                else:
                    evals[i] = r[0]
            except Exception:
                pass
        return (evals, evecs) if eigenvectors else evals

def eigh_min(F, X):
    H = hessian(F, X)
    evals, evecs = safe_eigh(H)
    return evals[:, 0], evals[:, -1], evecs[:, :, 0]

def lanczos_extremes(F, X, k=25, seed=0):
    n, d = X.shape
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g).to(X.device)
    v = v / v.norm(dim=1, keepdim=True)
    vprev = torch.zeros_like(v)
    beta = torch.zeros(n, device=X.device)
    alphas, betas = [], []
    for _ in range(k):
        w = hvp(F, X, v) - beta[:, None] * vprev
        alpha = (w * v).sum(1)
        alphas.append(alpha)
        w = w - alpha[:, None] * v
        beta = w.norm(dim=1)
        vprev = v
        safe = beta > 1e-10
        vnext = torch.where(
            safe[:, None],
            w / beta[:, None].clamp_min(1e-10),
            torch.randn(n, d, generator=g).to(X.device),
        )
        v = vnext / vnext.norm(dim=1, keepdim=True)
        betas.append(beta)
    A = torch.stack(alphas, 1)
    B = torch.stack(betas[:-1], 1)
    T = torch.diag_embed(A) + torch.diag_embed(B, 1) + torch.diag_embed(B, -1)
    ev = safe_eigh(T, eigenvectors=False)
    return ev[:, 0].cpu().numpy(), ev[:, -1].cpu().numpy()

def _refine(F, cand, L, d, keep, use_lanczos, max_try=None):
    def gf(p):
        x = torch.tensor(p)[None].requires_grad_(True)
        F(x).sum().backward()
        return x.grad[0].detach().cpu().numpy()

    found = []
    for p0 in (cand if max_try is None else cand[:max_try]):
        try:
            sol, info, ier, _ = optimize.fsolve(gf, p0, full_output=True)
        except Exception:
            continue
        if ier != 1 or np.abs(sol).max() > L or np.linalg.norm(gf(sol)) > 1e-6:
            continue
        st = torch.tensor(sol)[None]
        if use_lanczos:
            lmin, lmax = lanczos_extremes(F, st, k=min(30, d))
            lmin, lmax = lmin[0], lmax[0]
        else:
            lmn, lmx, _ = eigh_min(F, st)
            lmin, lmax = lmn.item(), lmx.item()
        if lmin < -1e-4 and lmax > 1e-4 and all(np.linalg.norm(sol - q) > 0.3 for q in found):
            found.append(sol)
    found.sort(key=lambda s: np.linalg.norm(s))
    return found[:keep]

def find_saddles_2d(F, L, grid=240, n_cand=800, keep=3):
    xs = np.linspace(-L, L, grid)
    G = torch.tensor(np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2))
    gn = grad(F, G).norm(dim=1).cpu().numpy()
    Gnp = G.cpu().numpy()
    order = np.lexsort((Gnp[:, 1], Gnp[:, 0], gn))
    return _refine(F, Gnp[order[:n_cand]], L, 2, keep, use_lanczos=False)

def find_saddles_nd(F, d, L, n_init=5000, n_cand=800, keep=3, seed=0, max_try=200):
    rng = np.random.default_rng(seed)
    Gnp = rng.uniform(-L, L, size=(n_init, d))
    gn = grad(F, torch.tensor(Gnp)).norm(dim=1).cpu().numpy()
    keys = tuple(Gnp[:, i] for i in range(d - 1, -1, -1)) + (gn,)
    order = np.lexsort(keys)
    return _refine(F, Gnp[order[:n_cand]], L, d, keep, use_lanczos=True, max_try=max_try)

OPTS = ['GD', 'Adam', 'RMSProp', 'AdaGrad', 'AdamW', 'SGD_mom']

def make_opt(name, P, lr):
    if name == 'GD':
        return torch.optim.SGD([P], lr=lr)
    if name == 'Adam':
        return torch.optim.Adam([P], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if name == 'RMSProp':
        return torch.optim.RMSprop([P], lr=lr, alpha=0.99, eps=1e-8)
    if name == 'AdaGrad':
        return torch.optim.Adagrad([P], lr=lr, eps=1e-8)
    if name == 'AdamW':
        return torch.optim.AdamW([P], lr=lr, weight_decay=0.01)
    if name == 'SGD_mom':
        return torch.optim.SGD([P], lr=lr, momentum=0.9)
    raise ValueError(name)

VAR = [
    ('A', 1.5), ('A', 2.0), ('A', 3.0),
    ('B', 1e-2), ('B', 1e-3), ('B', 1e-4),
    ('C', 0.5), ('C', 1.0), ('C', 2.0),
    ('D', 0.25), ('D', 0.5), ('D', 1.0),
]
HEAD = {'A': 2.0, 'B': 1e-3, 'C': 1.0, 'D': 0.5}

def headline_idx(fam):
    return VAR.index((fam, HEAD[fam]))

def run_config(F, s, v, r_curv, f_s, opt_name, lr, N, Tmax, seed, families, lambda_fn):
    torch.manual_seed(seed)
    X = (s[None] + 0.1 * torch.randn(N, s.shape[0], device=s.device)).requires_grad_(True)
    opt = make_opt(opt_name, X, lr)
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

def see_pt(e, s):
    return (e.mean() / s[e].mean()) if e.any() else 0.0

def see_ci(e, s, rng, B=2000):
    n = len(e)
    idx = rng.integers(0, n, (B, n))
    E, S = e[idx], s[idx]
    cnt = E.sum(1)
    tau = np.where(cnt > 0, np.where(E, S, 0).sum(1) / np.maximum(cnt, 1), np.nan)
    vals = np.where(cnt > 0, E.mean(1) / tau, 0.0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return see_pt(e, s), lo, hi

def kendall_w(mat):
    R = np.array([stats.rankdata(row) for row in mat])
    m, n = R.shape
    S = ((R.sum(0) - R.sum() / n) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))
