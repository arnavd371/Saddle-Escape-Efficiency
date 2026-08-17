# ==== bundled from see_core.py ====
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


# ==== bundled from optimizers.py ====
"""
Optimizer factory for the OPT 2026 extension.

Preserves the baseline's four optimizers exactly (GD_fixed, Adam, RMSProp,
AdaGrad -- same hyperparameters as Cell 1 of the notebook) and adds three
more so the paper can speak to the "Can Anything Beat Adam?" theme directly:

  - AdamW: decoupled weight decay (torch.optim.AdamW), weight_decay=1e-2.
    NOTE on weight_decay choice: these are toy 2D/nD saddle problems with no
    generalization concept (no train/test split, no overfitting to regularize
    against), so "weight decay" here has no principled loss-landscape meaning
    -- it merely adds a constant L2 pull toward the origin, `-wd*lr*x`, on top
    of the ordinary Adam update. We use PyTorch's own default value (1e-2) so
    the only difference we introduce vs. plain Adam is the decoupled decay
    term itself, not a hand-tuned magnitude. This is flagged explicitly in
    RESULTS_SUMMARY.md as a caveat on any AdamW-vs-Adam comparison, since the
    decay term will pull trajectories back toward the saddle a bit whenever
    the saddle is not at the origin (most of ours aren't).
  - SGD_Nesterov: torch.optim.SGD(momentum=0.9, nesterov=True).
  - Lion: no torch built-in, implemented manually below. Per the Lion paper
    (Chen et al. 2023), Lion's update uses sign(momentum-interpolated grad)
    scaled by lr, so it needs an lr roughly 3-10x SMALLER than Adam's for a
    comparable step size (sign() saturates the update magnitude to a
    fixed-norm step regardless of gradient scale, unlike Adam's per-coordinate
    normalization which is also roughly fixed-norm but with a different
    constant). We apply a 10x-smaller-lr convention: Lion's swept LR grid is
    LRS/10 elementwise, i.e. {1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2}, so that
    "same LR index" is comparable in effective step character across
    optimizers, and Lion's own table/figure axis is labeled with its actual
    (scaled) LR to avoid confusion.
"""
import torch


class Lion(torch.optim.Optimizer):
    """Manual implementation of Lion (EvoLved Sign Momentum), Chen et al. 2023.
    Defaults (beta1=0.9, beta2=0.99) match the paper's recommended defaults.
    Update rule per step:
        c_t = beta1 * m_{t-1} + (1-beta1) * g_t
        theta_t = theta_{t-1} - lr * ( sign(c_t) + wd * theta_{t-1} )
        m_t = beta2 * m_{t-1} + (1-beta2) * g_t
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                c = m.mul(beta1).add(grad, alpha=1 - beta1)
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)
                p.add_(torch.sign(c), alpha=-lr)
                m.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


LION_LR_SCALE = 0.1  # Lion's swept LRs = baseline LRS * this factor (paper convention: ~10x smaller than Adam)

# Optimizer set: legacy 4 unchanged (same call signature/hyperparams as the
# baseline notebook), plus 3 new ones for the extension.
OPTS_LEGACY = ['GD_fixed', 'Adam', 'RMSProp', 'AdaGrad']
OPTS_NEW = ['AdamW', 'SGD_Nesterov', 'Lion']
OPTS_EXT = OPTS_LEGACY + OPTS_NEW  # n=7


def make_opt(nm, P, lr):
    if nm == 'GD_fixed':
        return torch.optim.SGD([P], lr=lr)
    if nm == 'Adam':
        return torch.optim.Adam([P], lr=lr)
    if nm == 'RMSProp':
        return torch.optim.RMSprop([P], lr=lr, alpha=0.99)
    if nm == 'AdaGrad':
        return torch.optim.Adagrad([P], lr=lr)
    if nm == 'AdamW':
        return torch.optim.AdamW([P], lr=lr, weight_decay=1e-2)
    if nm == 'SGD_Nesterov':
        return torch.optim.SGD([P], lr=lr, momentum=0.9, nesterov=True)
    if nm == 'Lion':
        return Lion([P], lr=lr)
    raise ValueError(f"unknown optimizer {nm}")


def lr_for(nm, base_lr):
    """The LR actually used for a given optimizer at a given base-LR grid
    point. Only Lion rescales; everyone else uses the shared LRS grid as-is
    so baseline optimizers are byte-for-byte comparable to results_final/."""
    return base_lr * LION_LR_SCALE if nm == 'Lion' else base_lr


# ==== driver: ext1_optimizers ====
import time, os, pickle
import numpy as np, pandas as pd
import torch
from scipy import stats

def pick_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        t = torch.zeros(2, 2, device='cuda', dtype=torch.float64)
        (t @ t).sum().item()  # a real kernel launch, not just allocation
        return torch.device('cuda')
    except Exception as ex:
        print(f"[ext1] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext1] device = {device}   torch={torch.__version__}  numpy={np.__version__}  scipy={__import__('scipy').__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
RAWDIR = os.path.join(OUTDIR, 'raw')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. saddle verification (identical thresholds to baseline)
print('Locating and verifying saddles (n=7 optimizer extension, functions/thresholds unchanged from baseline)...')
SAD = {}; GEOM = {}
for nm, F in FUNCS_2D.items():
    t0 = time.time()
    ss = find_saddles_2d(F, DOM_2D[nm], device=device)
    SAD[nm] = ss
    if not ss:
        print(f'  {nm}: NONE -> excluded'); continue
    s = ss[0]
    lmin, lmax, Hxx, Hxy, Hyy = beigs(F, s[None])
    lmin_v = lmin.item()
    v = torch.tensor([lmin_v - Hyy.item(), Hxy.item()], device=device) if abs(Hxy.item()) > 1e-12 else torch.tensor([1., 0.], device=device)
    v = v / v.norm()
    r_curv = 1 / np.sqrt(abs(lmin_v))
    f_s = F(s[None])[0].item()
    GEOM[nm] = (s, v, r_curv, f_s, lmin_v)
    print(f'  {nm}: {len(ss)} saddle(s); using {s.cpu().numpy().round(3)}  lambda_min={lmin_v:.2f}  r_curv={r_curv:.3f}  ({time.time()-t0:.1f}s)')
FUNCS_2D_ACTIVE = {k: v for k, v in FUNCS_2D.items() if SAD.get(k)}

# ---------------------------------------------------------------- 2. trajectory generation, all 7 optimizers x 6 LR x 5 functions
print(f'\nRunning all configs: {len(FUNCS_2D_ACTIVE)} functions x {len(OPTS_EXT)} optimizers x {len(LRS)} LRs = '
      f'{len(FUNCS_2D_ACTIVE)*len(OPTS_EXT)*len(LRS)} configs')
DATA = {}
t_start = time.time()
for fn, F in FUNCS_2D_ACTIVE.items():
    for o in OPTS_EXT:
        for base_lr in LRS:
            actual_lr = lr_for(o, base_lr)
            e, s = run_config_2d(F, GEOM[fn], o, actual_lr, make_opt, device=device)
            # key by BASE lr (the shared grid index) so tables line up across optimizers,
            # even though Lion's *actual* lr differs -- actual_lr is stored alongside.
            DATA[(fn, o, base_lr)] = (e, s, actual_lr)
    print(f'  {fn} done. ({time.time()-t_start:.1f}s elapsed)')

# ---------------------------------------------------------------- 2b. Lion robustness check: unscaled LR grid
# Lion's headline results above use LRS*LION_LR_SCALE (paper convention: ~10x
# smaller than Adam). Its SEE came out uniformly low under that convention --
# before reporting "Lion doesn't pattern with RMSProp despite sign-based
# updates" as a finding, check whether that's an LR-scaling artifact by also
# running Lion at the SAME (unscaled) LRS grid as every other optimizer.
print('\nLion robustness check: rerunning at unscaled LRS grid (no 10x scaling)...')
DATA_LION_UNSCALED = {}
t_lion = time.time()
for fn, F in FUNCS_2D_ACTIVE.items():
    for base_lr in LRS:
        e, s = run_config_2d(F, GEOM[fn], 'Lion', base_lr, make_opt, device=device)  # actual_lr == base_lr, unscaled
        DATA_LION_UNSCALED[(fn, base_lr)] = (e, s)
print(f'  Lion unscaled-LR sweep done. ({time.time()-t_lion:.1f}s)')

lion_compare_rows = []
for fn in FUNCS_2D_ACTIVE:
    for fam in FAMS:
        k = headline_idx(fam)
        scaled_best = max(see_pt(DATA[(fn, 'Lion', lr)][0][k], DATA[(fn, 'Lion', lr)][1][k]) for lr in LRS)
        unscaled_best = max(see_pt(DATA_LION_UNSCALED[(fn, lr)][0][k], DATA_LION_UNSCALED[(fn, lr)][1][k]) for lr in LRS)
        lion_compare_rows.append({'function': fn, 'criterion': fam,
                                   'Lion_best_scaled_lr': scaled_best, 'Lion_best_unscaled_lr': unscaled_best,
                                   'diff': unscaled_best - scaled_best})
lion_df = pd.DataFrame(lion_compare_rows)
lion_df.to_csv(os.path.join(OUTDIR, 'lion_lr_robustness_check.csv'), index=False)
print(lion_df.to_string(index=False))
print(f"\nMean Lion SEE, scaled-LR convention:   {lion_df['Lion_best_scaled_lr'].mean():.3f}")
print(f"Mean Lion SEE, unscaled-LR grid:       {lion_df['Lion_best_unscaled_lr'].mean():.3f}")

# save raw trial data (esc/stp arrays) so tables can be regenerated without rerunning
with open(os.path.join(RAWDIR, 'ext1_raw_trials.pkl'), 'wb') as f:
    pickle.dump({'DATA': DATA, 'DATA_LION_UNSCALED': DATA_LION_UNSCALED, 'VAR': VAR,
                 'GEOM': {k: (v[0].cpu().numpy(), v[1].cpu().numpy(), v[2], v[3], v[4]) for k, v in GEOM.items()},
                 'OPTS_EXT': OPTS_EXT, 'LRS': LRS, 'SEED': SEED, 'N': N, 'TMAX': TMAX}, f)
print(f'Raw trial data saved to {RAWDIR}/ext1_raw_trials.pkl')

# ---------------------------------------------------------------- 3. SEE tables (mirrors notebook Cell 1 exactly, OPTS_EXT instead of OPTS)
rows = []
print('\nSEE at lr=0.2  (A_fixed-r / B_curvature / C_eigendisp / D_loss)  [n=7 optimizers]')
for fn in FUNCS_2D_ACTIVE:
    print(f'\n{fn}:')
    for o in OPTS_EXT:
        e, s, actual_lr = DATA[(fn, o, 0.2)]
        vals = []
        for fam in FAMS:
            k = headline_idx(fam)
            v, ci = see_ci(e[k], s[k], rng=rng)
            vals.append((v, ci))
        rows.append({'function': fn, 'optimizer': o, 'lr_grid': 0.2, 'lr_actual': actual_lr,
                      **{f'SEE_{f}': v for f, (v, _) in zip(FAMS, vals)},
                      **{f'CI_{f}': c for f, (_, c) in zip(FAMS, vals)}})
        print(f'  {o:13s}: ' + ' / '.join(f'{v:.3f}±{c:.3f}' for v, c in vals))
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, 'main_lr02_ext.csv'), index=False)

print('\nSEE at each optimizer\'s BEST LR per criterion  [n=7 optimizers]')
best_rows = []
for fn in FUNCS_2D_ACTIVE:
    print(f'\n{fn}:')
    for o in OPTS_EXT:
        vals = []
        for fam in FAMS:
            k = headline_idx(fam)
            v = max(see_pt(DATA[(fn, o, lr)][0][k], DATA[(fn, o, lr)][1][k]) for lr in LRS)
            vals.append(v)
        best_rows.append({'function': fn, 'optimizer': o, **{f'best_{f}': v for f, v in zip(FAMS, vals)}})
        print(f'  {o:13s}: ' + ' / '.join(f'{v:.3f}' for v in vals))
pd.DataFrame(best_rows).to_csv(os.path.join(OUTDIR, 'best_lr_ext.csv'), index=False)

# ---------------------------------------------------------------- 4. cross-criterion stats (n now 6-7 instead of 4 -- report n explicitly)
print(f'\n pairwise Spearman between criteria (best-LR ranks), n={len(OPTS_EXT)} optimizers ')
spearman_rows = []
for fn in FUNCS_2D_ACTIVE:
    sub = [r for r in best_rows if r['function'] == fn]
    a = {f: [r['best_' + f] for r in sub] for f in FAMS}
    for i, f1 in enumerate(FAMS):
        for j, f2 in enumerate(FAMS):
            if j <= i:
                continue
            rho, n_used = spearman_with_n(a[f1], a[f2])
            spearman_rows.append({'function': fn, 'pair': f'{f1}-{f2}', 'rho': rho, 'n': n_used})
    print(f"  {fn:11s} A-B:{stats.spearmanr(a['A'],a['B']).correlation:+.2f} "
          f"A-C:{stats.spearmanr(a['A'],a['C']).correlation:+.2f} "
          f"A-D:{stats.spearmanr(a['A'],a['D']).correlation:+.2f} "
          f"B-D:{stats.spearmanr(a['B'],a['D']).correlation:+.2f}   (n={len(OPTS_EXT)})")
pd.DataFrame(spearman_rows).to_csv(os.path.join(OUTDIR, 'spearman_pairwise_ext.csv'), index=False)

print(f"\nKendall's W concordance across the 4 criteria (best-LR), n={len(OPTS_EXT)} optimizers, with bootstrap CI")
w_rows = []
for fn in FUNCS_2D_ACTIVE:
    sub = [r for r in best_rows if r['function'] == fn]
    V = np.array([[r['best_' + f] for r in sub] for f in FAMS])  # (4 judges, n_opt items)
    res = kendalls_w_ci(V, rng=rng, method='bootstrap')
    w_rows.append({'function': fn, **res})
    print(f"  {fn:11s} W={res['W']:.2f}  95% CI [{res['ci_lo']:.2f}, {res['ci_hi']:.2f}]  (n_items={res['n_items']})")
pd.DataFrame(w_rows).to_csv(os.path.join(OUTDIR, 'kendall_w_ext.csv'), index=False)

print('\nwithin-family threshold stability (Spearman across variants, best-LR)')
stability_rows = []
for fam, ps in [('A', [1.5, 2.0, 3.0]), ('B', [1e-2, 1e-3, 1e-4]), ('C', [0.5, 1.0, 2.0]), ('D', [0.25, 0.5, 1.0])]:
    rhos = []
    for fn in FUNCS_2D_ACTIVE:
        rank = {}
        for p in ps:
            k = VAR.index((fam, p))
            rank[p] = [max(see_pt(DATA[(fn, o, lr)][0][k], DATA[(fn, o, lr)][1][k]) for lr in LRS) for o in OPTS_EXT]
        r1, n_used = spearman_with_n(rank[ps[0]], rank[ps[-1]])
        if not np.isnan(r1):
            rhos.append(r1)
            stability_rows.append({'family': fam, 'function': fn, 'rho': r1, 'n': n_used})
    print(f'  family {fam}: mean rho(extreme thresholds) = {np.mean(rhos):+.2f} over {len(rhos)} functions  (n={len(OPTS_EXT)} each)')
pd.DataFrame(stability_rows).to_csv(os.path.join(OUTDIR, 'within_family_stability_ext.csv'), index=False)

# ---------------------------------------------------------------- 5. AdamW-vs-Adam / Lion-vs-RMSProp qualitative pattern check
print('\nQualitative pattern check: does AdamW behave like Adam? Does Lion pattern with RMSProp?')
pattern_rows = []
for fn in FUNCS_2D_ACTIVE:
    sub = {r['optimizer']: r for r in best_rows if r['function'] == fn}
    for fam in FAMS:
        adam_v, adamw_v = sub['Adam']['best_' + fam], sub['AdamW']['best_' + fam]
        rms_v, lion_v = sub['RMSProp']['best_' + fam], sub['Lion']['best_' + fam]
        pattern_rows.append({'function': fn, 'criterion': fam,
                              'Adam': adam_v, 'AdamW': adamw_v, 'abs_diff_Adam_AdamW': abs(adam_v - adamw_v),
                              'RMSProp': rms_v, 'Lion': lion_v, 'abs_diff_RMSProp_Lion': abs(rms_v - lion_v)})
pd.DataFrame(pattern_rows).to_csv(os.path.join(OUTDIR, 'adamw_lion_pattern_check_ext.csv'), index=False)
print(pd.DataFrame(pattern_rows).groupby('criterion')[['abs_diff_Adam_AdamW', 'abs_diff_RMSProp_Lion']].mean().to_string())

print(f'\nExt1 done. Total elapsed: {time.time()-t_start:.1f}s. Files in {OUTDIR}/')
