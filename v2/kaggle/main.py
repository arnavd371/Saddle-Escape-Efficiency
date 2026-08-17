"""SEE pipeline. Single file for a Kaggle script kernel (T4).

Same logic as the v2/ modules. Push with NvidiaTeslaT4 — the default P100
is sm_60 and will not run this PyTorch build.
"""
import time, os, itertools
import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for d in ["out1", "out2", "out3", "figs1", "figs2", "figs3"]:
    os.makedirs(d, exist_ok=True)


torch.set_default_dtype(torch.float64)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    # Kaggle's default P100 (sm_60) advertises CUDA but the preinstalled
    # PyTorch wheel needs sm_70+. Probe with a real autograd round-trip
    # instead of trusting is_available().
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

# Rosenbrock is unimodal — no saddle, so a saddle-escape score is meaningless.
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
    # F(X).sum() decouples batch rows, so two autograd passes still give
    # per-row H@v rather than a cross-row mixture.
    X = X.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(F(X).sum(), X, create_graph=True)[0]
    return torch.autograd.grad((g * v).sum(), X)[0].detach()

def hessian(F, X):
    # d HVP calls. Fine for d=2; at d=50 use lanczos_extremes instead.
    n, d = X.shape
    cols = []
    for i in range(d):
        e = torch.zeros_like(X)
        e[:, i] = 1.
        cols.append(hvp(F, X, e))
    return torch.stack(cols, -1)

def safe_eigh(H, eigenvectors=True):
    # Diverged (clamped) points make Hessians that cuSOLVER refuses;
    # CPU LAPACK usually just returns garbage. Retry rows one at a time
    # and leave failures as NaN rather than killing the batch.
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
    # Smallest-eigenvalue eigenvector. Columns of torch.linalg.eigh, no 2x2 shortcut.
    H = hessian(F, X)
    evals, evecs = safe_eigh(H)
    return evals[:, 0], evals[:, -1], evecs[:, :, 0]

def lanczos_extremes(F, X, k=25, seed=0):
    # Extreme eigenvalues of the HVP operator. k steps vs d dense columns.
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
        # 1e-4 keeps numerically flat critical points out; 0.3 is a cheap
        # duplicate radius in the scaled domain.
        if lmin < -1e-4 and lmax > 1e-4 and all(np.linalg.norm(sol - q) > 0.3 for q in found):
            found.append(sol)
    found.sort(key=lambda s: np.linalg.norm(s))
    return found[:keep]

def find_saddles_2d(F, L, grid=240, n_cand=800, keep=3):
    xs = np.linspace(-L, L, grid)
    G = torch.tensor(np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2))
    gn = grad(F, G).norm(dim=1).cpu().numpy()
    Gnp = G.cpu().numpy()
    # gn primary; x then y as a lex tie-break so the same saddle is picked
    # on CPU and GPU, not whichever candidate numpy happens to sort first.
    order = np.lexsort((Gnp[:, 1], Gnp[:, 0], gn))
    return _refine(F, Gnp[order[:n_cand]], L, 2, keep, use_lanczos=False)

def find_saddles_nd(F, d, L, n_init=5000, n_cand=800, keep=3, seed=0, max_try=200):
    rng = np.random.default_rng(seed)
    Gnp = rng.uniform(-L, L, size=(n_init, d))
    gn = grad(F, torch.tensor(Gnp)).norm(dim=1).cpu().numpy()
    keys = tuple(Gnp[:, i] for i in range(d - 1, -1, -1)) + (gn,)
    order = np.lexsort(keys)
    # max_try caps fsolve+Lanczos: at d=50 each refinement is ~30 HVPs.
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
        return torch.optim.Adagrad([P], lr=lr, eps=1e-8)  # PyTorch default is 1e-10
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
        # 1e4 is already far past any escape threshold. 1e6 made Hessians
        # ill-conditioned enough for GPU eigh to throw.
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
                # r_curv = 1/sqrt(|lmin|) is the quadratic unstable-manifold scale
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
    # mat: (n_judges, n_items). rankdata averages ties.
    R = np.array([stats.rankdata(row) for row in mat])
    m, n = R.shape
    S = ((R.sum(0) - R.sum() / n) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))


D_PARAM = 2 * 8 + 8 * 1 + 1

def xor_data(seed=42, n_per=50, noise=0.05):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y_base = torch.tensor([0., 1., 1., 0.])
    X = base.repeat_interleave(n_per, 0) + noise * torch.randn(4 * n_per, 2, generator=g)
    y = y_base.repeat_interleave(n_per, 0)
    return X, y

def unflatten(theta):
    W1 = theta[:16].reshape(2, 8)
    W2 = theta[16:24].reshape(8, 1)
    b2 = theta[24:25]
    return W1, W2, b2

def make_loss_fn(X, y):
    def loss_one(theta):
        W1, W2, b2 = unflatten(theta)
        h = torch.tanh(X @ W1)
        out = (h @ W2 + b2).squeeze(-1)
        return ((out - y) ** 2).mean()
    return torch.func.vmap(loss_one)


OUTDIR, FIGDIR = 'out1', 'figs1'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

SEED, N, TMAX = 42, 200, 200
LRS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
FAMS = ['A', 'B', 'C', 'D']
rng = np.random.default_rng(SEED)
t0 = time.time()

def lambda_fn_2d(F):
    def f(Xd):
        H = hessian(F, Xd)
        return safe_eigh(H, eigenvectors=False).cpu().numpy()[:, 0]
    return f

GEOM = {}
for name, (F, L) in FUNCS2D.items():
    ss = find_saddles_2d(F, L)
    if not ss:
        print(f'{name}: no saddle found (excluded)')
        continue
    s = torch.tensor(ss[0])
    lmin, lmax, v = eigh_min(F, s[None])
    lmin = lmin.item()
    v = v[0] / v[0].norm()
    r_curv = 1 / np.sqrt(abs(lmin))
    f_s = F(s[None])[0].item()
    s, v = s.to(DEVICE), v.to(DEVICE)
    GEOM[name] = dict(s=s, v=v, r_curv=r_curv, f_s=f_s, lmin=lmin, n_saddles=len(ss))
    print(f'{name}: {len(ss)} saddle(s), s={s.cpu().numpy().round(3)}, lmin={lmin:.2f}, r_curv={r_curv:.3f}')
print('saddle search', round(time.time() - t0, 1))

DATA = {}
for name in GEOM:
    F, L = FUNCS2D[name]
    g, lf = GEOM[name], lambda_fn_2d(F)
    for o in OPTS:
        for lr in LRS:
            DATA[(name, o, lr)] = run_config(
                F, g['s'], g['v'], g['r_curv'], g['f_s'], o, lr, N, TMAX, SEED, FAMS, lf)
    print(name, 'done', round(time.time() - t0, 1))

rows = []
for name in GEOM:
    for o in OPTS:
        e, s = DATA[(name, o, 0.2)]
        rec = {'function': name, 'optimizer': o}
        for fam in FAMS:
            k = headline_idx(fam)
            val, lo, hi = see_ci(e[k], s[k], rng)
            rec[f'SEE_{fam}'] = val
            rec[f'CI_lo_{fam}'] = lo
            rec[f'CI_hi_{fam}'] = hi
        rows.append(rec)
pd.DataFrame(rows).to_csv(f'{OUTDIR}/main_lr02.csv', index=False)

best_rows = []
for name in GEOM:
    for o in OPTS:
        rec = {'function': name, 'optimizer': o}
        for fam in FAMS:
            k = headline_idx(fam)
            rec[f'best_{fam}'] = max(
                see_pt(DATA[(name, o, lr)][0][k], DATA[(name, o, lr)][1][k]) for lr in LRS)
        best_rows.append(rec)
pd.DataFrame(best_rows).to_csv(f'{OUTDIR}/best_lr.csv', index=False)

pairs = [(a, b) for i, a in enumerate(FAMS) for b in FAMS[i + 1:]]
spear_rows, kw_rows = [], []
for name in GEOM:
    sub = [r for r in best_rows if r['function'] == name]
    mat = np.array([[r[f'best_{f}'] for f in FAMS] for r in sub])
    for a, b in pairs:
        ia, ib = FAMS.index(a), FAMS.index(b)
        rho = stats.spearmanr(mat[:, ia], mat[:, ib]).correlation
        spear_rows.append({'function': name, 'pair': f'{a}-{b}', 'rho': rho})
    kw_rows.append({'function': name, 'W': kendall_w(mat.T), 'lmin': GEOM[name]['lmin']})
pd.DataFrame(spear_rows).to_csv(f'{OUTDIR}/spearman.csv', index=False)
pd.DataFrame(kw_rows).to_csv(f'{OUTDIR}/kendall_w.csv', index=False)

stab_rows = []
for fam, ps in [('A', [1.5, 2.0, 3.0]), ('B', [1e-2, 1e-3, 1e-4]),
                ('C', [0.5, 1.0, 2.0]), ('D', [0.25, 0.5, 1.0])]:
    for name in GEOM:
        rank = {}
        for p in ps:
            k = VAR.index((fam, p))
            rank[p] = [max(see_pt(DATA[(name, o, lr)][0][k], DATA[(name, o, lr)][1][k])
                           for lr in LRS) for o in OPTS]
        rho = stats.spearmanr(rank[ps[0]], rank[ps[-1]]).correlation
        stab_rows.append({'family': fam, 'function': name, 'rho': rho})
pd.DataFrame(stab_rows).to_csv(f'{OUTDIR}/within_family.csv', index=False)

print('printed tables')
for name in GEOM:
    print(name)
    for o in OPTS:
        r = [x for x in best_rows if x['function'] == name and x['optimizer'] == o][0]
        print(f"  {o:9s} " + ' '.join(f'{f}:{r[f"best_{f}"]:.3f}' for f in FAMS))

names = list(GEOM.keys())
fig, axes = plt.subplots(len(names), 4, figsize=(14, 2.1 * len(names)), squeeze=False)
crit_names = {'A': 'fixed radius', 'B': 'curvature', 'C': 'eigen-disp', 'D': 'loss-drop'}
colors = dict(zip(OPTS, ['#4477aa', '#ee7733', '#228833', '#cc3311', '#aa3377', '#66ccee']))
for i, name in enumerate(names):
    sub = [r for r in best_rows if r['function'] == name]
    for j, fam in enumerate(FAMS):
        ax = axes[i, j]
        vals = [r[f'best_{fam}'] for r in sub]
        ax.bar(range(len(OPTS)), vals, color=[colors[o] for o in OPTS])
        ax.set_xticks(range(len(OPTS)))
        ax.set_xticklabels(OPTS, fontsize=6, rotation=45)
        ax.set_ylim(0, 1.05)
        if j == 0:
            ax.set_ylabel(name, fontsize=8)
        if i == 0:
            ax.set_title(crit_names[fam], fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig1_criteria_grid.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for idx, name in enumerate(names):
    ax = axes[idx // 4, idx % 4]
    M = np.eye(4)
    for a, b in pairs:
        r = [x for x in spear_rows if x['function'] == name and x['pair'] == f'{a}-{b}'][0]['rho']
        ia, ib = FAMS.index(a), FAMS.index(b)
        M[ia, ib] = M[ib, ia] = r
    ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(FAMS)
    ax.set_yticklabels(FAMS)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f'{M[a, b]:+.2f}', ha='center', va='center', fontsize=7)
    ax.set_title(name, fontsize=9)
for idx in range(len(names), 8):
    axes[idx // 4, idx % 4].axis('off')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig2_spearman.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.barh(names, [r['W'] for r in kw_rows], color='#4477aa')
ax.set_xlim(0, 1)
ax.set_xlabel("Kendall's W")
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig3_kendall_w.png', dpi=300, bbox_inches='tight')
plt.close()
print('exp1 total', round(time.time() - t0, 1), 's')


OUTDIR, FIGDIR = 'out2', 'figs2'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

SEED, N, TMAX = 42, 100, 500
LRS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
FAMS = ['A', 'B']
t0 = time.time()

FUNCSND = {'Rastrigin': (f_rastrigin, 5.12), 'Ackley': (f_ackley, 5.0)}
DIMS = [10, 50]

def lambda_fn_nd(F, d):
    def f(Xd):
        lmin, _ = lanczos_extremes(F, Xd, k=min(30, d))
        return lmin
    return f

GEOM = {}
for name, (F, L) in FUNCSND.items():
    for d in DIMS:
        ss = find_saddles_nd(F, d, L, seed=SEED)
        if not ss:
            print(f'{name} d={d}: no saddle found (excluded)')
            continue
        s = torch.tensor(ss[0])
        lmin, lmax = lanczos_extremes(F, s[None], k=min(30, d))
        s = s.to(DEVICE)
        GEOM[(name, d)] = dict(s=s, lmin=lmin[0])
        print(f'{name} d={d}: {len(ss)} saddle(s), lmin={lmin[0]:.2f}')
print('saddle search', round(time.time() - t0, 1))

DATA = {}
for (name, d) in GEOM:
    F, L = FUNCSND[name]
    g, lf = GEOM[(name, d)], lambda_fn_nd(F, d)
    for o in OPTS:
        for lr in LRS:
            DATA[(name, d, o, lr)] = run_config(
                F, g['s'], None, None, None, o, lr, N, TMAX, SEED, FAMS, lf)
    print(name, d, 'done', round(time.time() - t0, 1))

best_rows = []
for (name, d) in GEOM:
    for o in OPTS:
        rec = {'function': name, 'dim': d, 'optimizer': o}
        for fam in FAMS:
            k = headline_idx(fam)
            rec[f'best_{fam}'] = max(
                see_pt(DATA[(name, d, o, lr)][0][k], DATA[(name, d, o, lr)][1][k]) for lr in LRS)
        best_rows.append(rec)
pd.DataFrame(best_rows).to_csv(f'{OUTDIR}/best_lr.csv', index=False)

rho_rows = []
for (name, d) in GEOM:
    sub = [r for r in best_rows if r['function'] == name and r['dim'] == d]
    a = [r['best_A'] for r in sub]
    b = [r['best_B'] for r in sub]
    rho = stats.spearmanr(a, b).correlation
    rho_rows.append({'function': name, 'dim': d, 'rho_AB': rho})
    print(name, d, 'rho(A,B)=', rho)

anchor = 'out1/spearman.csv'
if os.path.exists(anchor):
    sp1 = pd.read_csv(anchor)
    for name in FUNCSND:
        row = sp1[(sp1['function'] == name) & (sp1['pair'] == 'A-B')]
        if len(row):
            rho_rows.append({'function': name, 'dim': 2, 'rho_AB': row.iloc[0]['rho']})
pd.DataFrame(rho_rows).to_csv(f'{OUTDIR}/rho_vs_dim.csv', index=False)

fig, ax = plt.subplots(figsize=(5, 3.5))
for name, c in [('Rastrigin', '#4477aa'), ('Ackley', '#ee7733')]:
    sub = sorted([r for r in rho_rows if r['function'] == name], key=lambda r: r['dim'])
    if sub:
        ax.plot([r['dim'] for r in sub], [r['rho_AB'] for r in sub], 'o-', color=c, label=name)
ax.set_xscale('log')
ax.set_xlabel('dimension')
ax.set_ylabel('Spearman rho(A,B)')
ax.axhline(0, color='gray', linewidth=0.7)
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/rho_ab_vs_dim.png', dpi=300, bbox_inches='tight')
plt.close()
print('exp2 total', round(time.time() - t0, 1))


OUTDIR, FIGDIR = 'out3', 'figs3'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

SEED, N, TMAX = 42, 50, 300
LRS = [0.001, 0.005, 0.01, 0.05, 0.1]
FAMS = ['A', 'B']
DOM_L = 4.0  # covers the saturated-tanh region where random inits land
t0 = time.time()

X, y = xor_data(seed=SEED)
F_cpu = make_loss_fn(X, y)
F = make_loss_fn(X.to(DEVICE), y.to(DEVICE))

ss = find_saddles_nd(F_cpu, D_PARAM, DOM_L, seed=SEED)
print('saddles found', len(ss), 'time', round(time.time() - t0, 1))
s = torch.tensor(ss[0])
lmin, lmax = lanczos_extremes(F_cpu, s[None], k=25)
print('using saddle, loss=', F_cpu(s[None]).item(), 'lmin=', lmin[0], 'lmax=', lmax[0])
s = s.to(DEVICE)

def lambda_fn(Xd):
    lm, _ = lanczos_extremes(F, Xd, k=25)
    return lm

DATA = {}
for o in OPTS:
    for lr in LRS:
        DATA[(o, lr)] = run_config(F, s, None, None, None, o, lr, N, TMAX, SEED, FAMS, lambda_fn)
    print(o, 'done', round(time.time() - t0, 1))

best_rows = []
for o in OPTS:
    rec = {'optimizer': o}
    for fam in FAMS:
        k = headline_idx(fam)
        rec[f'best_{fam}'] = max(
            see_pt(DATA[(o, lr)][0][k], DATA[(o, lr)][1][k]) for lr in LRS)
    best_rows.append(rec)
pd.DataFrame(best_rows).to_csv(f'{OUTDIR}/best_lr.csv', index=False)

a = [r['best_A'] for r in best_rows]
b = [r['best_B'] for r in best_rows]
rho = stats.spearmanr(a, b).correlation
print('NN rho(A,B)=', rho)
pd.DataFrame([{'rho_AB': rho, 'lmin_saddle': lmin[0]}]).to_csv(f'{OUTDIR}/rho_ab.csv', index=False)
for r in best_rows:
    print(f"{r['optimizer']:9s} A:{r['best_A']:.3f} B:{r['best_B']:.3f}")

fig, ax = plt.subplots(figsize=(5, 3.5))
xw = np.arange(len(OPTS))
ax.bar(xw - 0.15, [r['best_A'] for r in best_rows], width=0.3, label='A (distance)', color='#4477aa')
ax.bar(xw + 0.15, [r['best_B'] for r in best_rows], width=0.3, label='B (curvature)', color='#ee7733')
ax.set_xticks(xw)
ax.set_xticklabels(OPTS, rotation=45, fontsize=8)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title(f'XOR-MLP saddle, rho(A,B)={rho:+.2f}')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/nn_AB.png', dpi=300, bbox_inches='tight')
plt.close()
print('exp3 total', round(time.time() - t0, 1))

ORIGINAL5 = ['Himmelblau', 'Ackley', 'Rastrigin', 'Styblinski', 'Levy']
NEW3 = ['Beale', 'Booth', 'Schwefel']

def exact_spearman_p(x, y):
    # n=5: enumerate the permutation null rather than scipy's t approximation.
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rho0 = stats.spearmanr(x, y).correlation
    count = total = 0
    for perm in itertools.permutations(range(n)):
        r = stats.spearmanr(x, y[list(perm)]).correlation
        if abs(r) >= abs(rho0) - 1e-12:
            count += 1
        total += 1
    return rho0, count / total

kw = pd.read_csv('out1/kendall_w.csv').set_index('function')
sub = kw.loc[ORIGINAL5]
rho, p = exact_spearman_p(sub['lmin'].abs().values, sub['W'].values)
print('curvature-sharpness: rho(|lmin|,W) =', round(rho, 3), 'exact p =', round(p, 4), 'n=5')
sub.to_csv('out1/curvature_sharpness_input.csv')

print()
print('A-vs-B rank inversion across scales')
sp1 = pd.read_csv('out1/spearman.csv')
for name in ORIGINAL5 + NEW3:
    row = sp1[(sp1['function'] == name) & (sp1['pair'] == 'A-B')]
    if len(row):
        print(f'  2D {name:11s} rho(A,B)={row.iloc[0]["rho"]:+.2f}')

try:
    r2 = pd.read_csv('out2/rho_vs_dim.csv')
    for _, row in r2.sort_values(['function', 'dim']).iterrows():
        print(f"  {row['function']} d={row['dim']:<3.0f} rho(A,B)={row['rho_AB']:+.2f}")
except FileNotFoundError:
    print('  exp2 not finished yet')

try:
    r3 = pd.read_csv('out3/rho_ab.csv')
    print(f"  NN (XOR-MLP) rho(A,B)={r3.iloc[0]['rho_AB']:+.2f}")
except FileNotFoundError:
    print('  exp3 not finished yet')

print()
print('AdamW / SGD+momentum under A vs B')
best1 = pd.read_csv('out1/best_lr.csv')
for opt in ['AdamW', 'SGD_mom']:
    sub = best1[best1['optimizer'] == opt]
    print(f'  {opt}: mean best_A={sub["best_A"].mean():.3f}  mean best_B={sub["best_B"].mean():.3f}')
    for _, row in sub.iterrows():
        rankA = (best1[best1['function'] == row['function']]['best_A'] > row['best_A']).sum() + 1
        rankB = (best1[best1['function'] == row['function']]['best_B'] > row['best_B']).sum() + 1
        print(f"    {row['function']:11s} rank_A={rankA}/6  rank_B={rankB}/6")

print()
print('Criterion C saturation')
for name in ORIGINAL5 + NEW3:
    sub = best1[best1['function'] == name]
    if sub.empty or 'best_C' not in sub.columns:
        print(f'  {name:11s} excluded (no saddle found)')
        continue
    print(f"  {name:11s} best_C range [{sub['best_C'].min():.3f}, {sub['best_C'].max():.3f}]  "
          f"n_at_1.0={(sub['best_C'] > 0.999).sum()}/6")

print()
print('new benchmarks vs original five')
print(pd.read_csv('out1/kendall_w.csv').to_string(index=False))
