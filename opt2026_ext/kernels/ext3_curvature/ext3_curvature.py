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


# ==== bundled from curvature_family.py ====
"""
Parametrized saddle family for the continuous curvature-sharpness sweep (Ext 3),
replacing the informal 5-point Rastrigin scatter story with a controlled,
analytically-verified sweep over a single "oscillation density" parameter k.

    f(x, y; k) = a*x^2 - b*y^2 + k * sin(omega*x) * sin(omega*y)

Why this form is verifiable analytically:
  grad f(0,0) = (k*omega*cos(0)*sin(0), -2b*0 + k*omega*sin(0)*cos(0)) = (0, 0)
  for EVERY k -- the origin is a critical point regardless of the perturbation
  strength, because sin(0)=0 kills the cross term's gradient contribution at
  the origin exactly.

  Hessian at (0,0):
      H = [[2a,        k*omega^2],
           [k*omega^2, -2b      ]]
  det(H) = -4ab - (k*omega^2)^2, which is < 0 for ALL k since a,b>0 -- so
  (0,0) is a saddle point (indefinite Hessian) for every k in the sweep, not
  just a limited range. This is verified both analytically (closed form
  below) and numerically (cross-checked against the same finite-difference
  beigs() used everywhere else in the pipeline, and against the same
  acceptance thresholds as the 2D/nD saddle finders: ||grad||<1e-6,
  lambda_min<-1e-4, lambda_max>1e-4).

Parameter choices:
  a = b = 1.0 (a simple, symmetric base saddle -- no special curvature bias).
  omega = 2*pi, i.e. the same spatial period (1.0) as Rastrigin's own
    oscillatory term (Rastrigin: -10*cos(2*pi*x)), so k's sweep range can be
    calibrated directly against Rastrigin's oscillation density.
  k in [0, K_MAX]: at k=0 this is a perfectly smooth quadratic saddle (the
    "textbook" case where all four criteria should trivially agree). K_MAX is
    chosen so the perturbation's contribution to the Hessian at the origin,
    k*omega^2, reaches the same order of magnitude as Rastrigin's own
    curvature at its verified saddle (lambda_min approx -10*(2*pi)^2 ~ -394,
    confirmed empirically in Ext 1's saddle-finding log: Rastrigin lambda_min
    = -392.73). Solving k*omega^2 ~ 394 with omega=2*pi gives k ~ 10, so we
    sweep k in {0, ..., 10} -- "0 (perfectly smooth) to a value comparable in
    oscillatory character to Rastrigin," as specified.
"""
import numpy as np
import torch

A_COEF = 1.0
B_COEF = 1.0
OMEGA = 2 * np.pi
DOM_CURV = 5.0
K_VALUES = [round(v, 4) for v in np.linspace(0.0, 10.0, 18)]  # 18 values, 0..~Rastrigin-comparable


def make_F_curv(k):
    def F(X):
        x, y = X[:, 0], X[:, 1]
        return A_COEF * x ** 2 - B_COEF * y ** 2 + k * torch.sin(OMEGA * x) * torch.sin(OMEGA * y)
    return F


def analytic_hessian_at_origin(k):
    Hxx = 2 * A_COEF
    Hyy = -2 * B_COEF
    Hxy = k * OMEGA ** 2
    m = (Hxx + Hyy) / 2
    d = np.sqrt(((Hxx - Hyy) / 2) ** 2 + Hxy ** 2)
    return m - d, m + d, Hxx, Hxy, Hyy  # lmin, lmax, Hxx, Hxy, Hyy


def verify_and_build_geom(k, device='cpu'):
    """Verify (0,0) meets the SAME acceptance thresholds as every other
    saddle finder in this project, using both the closed-form analytic
    Hessian and the same finite-difference beigs() used elsewhere (agreement
    between the two is itself a correctness check on beigs' h=1e-4 step).
    Returns (ok: bool, geom tuple compatible with run_config_2d, diagnostics dict).
    """
    F = make_F_curv(k)
    s = torch.zeros(2, device=device)
    g = bgrad(F, s[None])
    grad_norm = g.norm().item()

    lmin_a, lmax_a, Hxx, Hxy, Hyy = analytic_hessian_at_origin(k)
    lmin_fd, lmax_fd, Hxx_fd, Hxy_fd, Hyy_fd = beigs(F, s[None])
    lmin_fd, lmax_fd = lmin_fd.item(), lmax_fd.item()

    ok = (grad_norm < 1e-6) and (lmin_a < -1e-4) and (lmax_a > 1e-4)

    v = torch.tensor([lmin_a - Hyy, Hxy], device=device) if abs(Hxy) > 1e-12 else torch.tensor([1., 0.], device=device)
    v = v / v.norm()
    r_curv = 1 / np.sqrt(abs(lmin_a))
    f_s = F(s[None])[0].item()
    geom = (s, v, r_curv, f_s, lmin_a)

    diagnostics = {'k': k, 'grad_norm': grad_norm,
                    'lmin_analytic': lmin_a, 'lmax_analytic': lmax_a,
                    'lmin_finite_diff': lmin_fd, 'lmax_finite_diff': lmax_fd,
                    'analytic_vs_fd_lmin_diff': abs(lmin_a - lmin_fd),
                    'analytic_vs_fd_lmax_diff': abs(lmax_a - lmax_fd)}
    return ok, geom, diagnostics


# ==== driver: ext3_curvature ====
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
        print(f"[ext3] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext3] device={device}  torch={torch.__version__}  numpy={np.__version__}  scipy={__import__('scipy').__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
RAWDIR = os.path.join(OUTDIR, 'raw')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)
rng = np.random.default_rng(SEED)

print(f'\nContinuous curvature sweep: {len(K_VALUES)} k-values from {K_VALUES[0]} to {K_VALUES[-1]}, '
      f'{len(OPTS_EXT)} optimizers x {len(LRS)} LRs = {len(K_VALUES)*len(OPTS_EXT)*len(LRS)} configs')

# ---------------------------------------------------------------- 1. verify saddle at origin for every k
print('\nVerifying saddle at (0,0) for every k (analytic + finite-difference cross-check)...')
verify_rows = []
GEOM_K = {}
for k in K_VALUES:
    ok, geom, diag = verify_and_build_geom(k, device=device)
    GEOM_K[k] = geom
    verify_rows.append({'k': k, 'accepted': ok, **diag})
    if not ok:
        print(f'  k={k:.3f}: FAILED acceptance thresholds -- {diag}')
pd.DataFrame(verify_rows).to_csv(os.path.join(OUTDIR, 'curvature_saddle_verification.csv'), index=False)
n_failed = sum(1 for r in verify_rows if not r['accepted'])
print(f'  {len(K_VALUES)-n_failed}/{len(K_VALUES)} k-values verified as saddles at (0,0). '
      f'(analytic-vs-finite-diff lmin agreement: max diff = {max(r["analytic_vs_fd_lmin_diff"] for r in verify_rows):.2e})')
K_VALUES_OK = [r['k'] for r in verify_rows if r['accepted']]

# ---------------------------------------------------------------- 2. trajectory generation across k x optimizer x LR
DATA = {}
t_start = time.time()
for k in K_VALUES_OK:
    F = make_F_curv(k)
    geom = GEOM_K[k]
    for o in OPTS_EXT:
        for base_lr in LRS:
            actual_lr = lr_for(o, base_lr)
            e, s = run_config_2d(F, geom, o, actual_lr, make_opt, device=device)
            DATA[(k, o, base_lr)] = (e, s, actual_lr)
    print(f'  k={k:.3f} done. ({time.time()-t_start:.1f}s elapsed)')

with open(os.path.join(RAWDIR, 'ext3_raw_trials.pkl'), 'wb') as f:
    pickle.dump({'DATA': DATA, 'VAR': VAR, 'K_VALUES': K_VALUES_OK,
                 'OPTS_EXT': OPTS_EXT, 'LRS': LRS, 'SEED': SEED, 'N': N, 'TMAX': TMAX,
                 'A_COEF': A_COEF, 'B_COEF': B_COEF, 'OMEGA': OMEGA}, f)
print(f'Raw trial data saved to {RAWDIR}/ext3_raw_trials.pkl')

# ---------------------------------------------------------------- 3. best-LR SEE per (k, optimizer, criterion)
best_rows = []
for k in K_VALUES_OK:
    for o in OPTS_EXT:
        vals = []
        for fam in FAMS:
            fi = headline_idx(fam)
            v = max(see_pt(DATA[(k, o, lr)][0][fi], DATA[(k, o, lr)][1][fi]) for lr in LRS)
            vals.append(v)
        best_rows.append({'k': k, 'optimizer': o, **{f'best_{f}': v for f, v in zip(FAMS, vals)}})
pd.DataFrame(best_rows).to_csv(os.path.join(OUTDIR, 'curvature_best_lr.csv'), index=False)

# ---------------------------------------------------------------- 4. Kendall's W (with bootstrap CI) vs k -- the headline curve
print("\nKendall's W across the 4 criteria, per k, with bootstrap CI")
w_rows = []
for k in K_VALUES_OK:
    sub = [r for r in best_rows if r['k'] == k]
    V = np.array([[r['best_' + f] for r in sub] for f in FAMS])  # (4 judges, n_opt items)
    res = kendalls_w_ci(V, rng=rng, method='bootstrap')
    w_rows.append({'k': k, **res})
    print(f'  k={k:6.3f}  W={res["W"]:.3f}  95% CI [{res["ci_lo"]:.3f}, {res["ci_hi"]:.3f}]  (n_items={res["n_items"]})')
w_df = pd.DataFrame(w_rows)
w_df.to_csv(os.path.join(OUTDIR, 'curvature_kendall_w_vs_k.csv'), index=False)

# ---------------------------------------------------------------- 5. correlation between k and W, with bootstrap CI
print('\nCorrelation between oscillation-density parameter k and Kendall\'s W')
ks = w_df['k'].values
Ws = w_df['W'].values
rho_obs, n_used = spearman_with_n(ks, Ws)
B = 2000
boot_rhos = []
idx_all = np.arange(len(ks))
for _ in range(B):
    idx = rng.integers(0, len(ks), len(ks))
    r = stats.spearmanr(ks[idx], Ws[idx]).correlation
    if not np.isnan(r):
        boot_rhos.append(r)
boot_rhos = np.array(boot_rhos)
ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
print(f'  Spearman rho(k, W) = {rho_obs:+.3f}  (n={n_used} k-values)  95% bootstrap CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')
with open(os.path.join(OUTDIR, 'curvature_k_vs_W_correlation.csv'), 'w') as f:
    f.write('rho,n,ci_lo,ci_hi,B\n')
    f.write(f'{rho_obs},{n_used},{ci_lo},{ci_hi},{B}\n')

print(f'\nExt3 done. Total elapsed: {time.time()-t_start:.1f}s. Files in {OUTDIR}/')
