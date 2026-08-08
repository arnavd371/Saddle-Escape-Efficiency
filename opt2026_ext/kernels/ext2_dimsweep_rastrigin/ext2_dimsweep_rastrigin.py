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


# ==== bundled from hessian_eig_nd.py ====
"""
N-dimensional generalization of see_core.beigs.

beigs() in see_core.py is a closed-form 2x2 finite-difference formula and
does not generalize past n=2. For arbitrary n we cannot afford a dense
Hessian (O(n^2) gradient evals per point, per step, per trial -- intractable
at n=50 x N=200 trials x T=200 steps x 630 configs). Instead we estimate just
lambda_min and lambda_max (all criterion B/D need) via batched Lanczos
tridiagonalization driven by Hessian-vector products (HVP), which costs
O(m) HVPs per point for m Lanczos iterations, independent of a dense O(n^2)
Hessian.

Correctness note on batching: our benchmark functions are row-separable,
i.e. F(X).sum() = sum_i f(X_i) with no cross terms between batch rows. That
lets a single double-backward pass compute a *batched* HVP -- one
Hessian-vector product per row, simultaneously -- via the same sum-trick
`bgrad` already uses for batched gradients. See `batched_hvp` below.
"""
import torch


def batched_hvp(F, X, V):
    """X: (N,n) requires no pre-existing grad state (we set it up here).
    V: (N,n) per-row direction vectors.
    Returns (N,n): Hv[i] = Hessian(f)(X[i]) @ V[i]. Exact (autodiff), not
    finite-difference -- no h-parameter to tune."""
    Xc = X.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(F(Xc).sum(), Xc, create_graph=True)[0]  # (N,n), per-row correct since F is row-separable
    gv = (g * V).sum()
    Hv = torch.autograd.grad(gv, Xc, retain_graph=False)[0]  # (N,n)
    return Hv.detach()


def batched_lanczos_extreme_eigs(F, X, m=30, device='cpu', eps=1e-12):
    """Batched Lanczos tridiagonalization to estimate lambda_min/lambda_max
    of the Hessian of F at each row of X, via m HVPs per row (no dense
    Hessian ever formed). Returns (lmin, lmax): each (N,).

    m controls accuracy: for well-separated extreme eigenvalues, m~20-40
    typically converges to 3+ significant figures. We default to 30, matching
    the accuracy/cost tradeoff documented in the diagnostic benchmark.
    """
    N, n = X.shape
    m = min(m, n)  # Lanczos can't exceed the dimension
    Xc = X.detach().clone()
    v = torch.randn(N, n, device=device, dtype=X.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)
    v_prev = torch.zeros_like(v)
    beta_prev = torch.zeros(N, device=device, dtype=X.dtype)

    alphas = []
    betas = []  # betas[j] connects Lanczos vector j and j+1; length m-1 used
    for j in range(m):
        w = batched_hvp(F, Xc, v)
        alpha = (w * v).sum(dim=1)
        alphas.append(alpha)
        w = w - alpha.unsqueeze(1) * v - beta_prev.unsqueeze(1) * v_prev
        beta = w.norm(dim=1)
        betas.append(beta)
        v_prev = v
        beta_safe = beta.clamp_min(eps)
        v = w / beta_safe.unsqueeze(1)
        beta_prev = beta

    A = torch.stack(alphas, dim=1)          # (N, m)
    B = torch.stack(betas[:-1], dim=1) if m > 1 else torch.zeros(N, 0, device=device, dtype=X.dtype)  # (N, m-1)

    T = torch.diag_embed(A)                  # (N, m, m)
    if m > 1:
        idx = torch.arange(m - 1, device=device)
        T[:, idx, idx + 1] = B
        T[:, idx + 1, idx] = B

    eigvals = torch.linalg.eigvalsh(T)        # (N, m) ascending
    return eigvals[:, 0], eigvals[:, -1]


def batched_lanczos_min_eigpair(F, X, m=30, device='cpu', eps=1e-12):
    """Like batched_lanczos_extreme_eigs, but also returns the (approximate)
    unit eigenvector of lambda_min per row -- needed for criterion C
    (eigenvector-projection escape test), which is undefined without it.
    Returns (lmin, lmax, vmin): (N,), (N,), (N,n).

    The eigenvector is recovered from the Lanczos Ritz pair: if y is the
    eigenvector of the small (m,m) tridiagonal T for its smallest eigenvalue,
    then V @ y (V = the (n,m) matrix of stored Lanczos basis vectors) is the
    Ritz vector approximation to the full-space eigenvector -- standard
    Lanczos post-processing, no extra HVPs required beyond what
    batched_lanczos_extreme_eigs already does.
    """
    N, n = X.shape
    m = min(m, n)
    Xc = X.detach().clone()
    v = torch.randn(N, n, device=device, dtype=X.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)
    v_prev = torch.zeros_like(v)
    beta_prev = torch.zeros(N, device=device, dtype=X.dtype)

    alphas = []
    betas = []
    V_stack = []  # store each Lanczos basis vector to reconstruct the eigenvector
    for j in range(m):
        V_stack.append(v)
        w = batched_hvp(F, Xc, v)
        alpha = (w * v).sum(dim=1)
        alphas.append(alpha)
        w = w - alpha.unsqueeze(1) * v - beta_prev.unsqueeze(1) * v_prev
        beta = w.norm(dim=1)
        betas.append(beta)
        v_prev = v
        beta_safe = beta.clamp_min(eps)
        v = w / beta_safe.unsqueeze(1)
        beta_prev = beta

    A = torch.stack(alphas, dim=1)
    B = torch.stack(betas[:-1], dim=1) if m > 1 else torch.zeros(N, 0, device=device, dtype=X.dtype)
    V = torch.stack(V_stack, dim=1)  # (N, m, n)

    T = torch.diag_embed(A)
    if m > 1:
        idx = torch.arange(m - 1, device=device)
        T[:, idx, idx + 1] = B
        T[:, idx + 1, idx] = B

    eigvals, eigvecs = torch.linalg.eigh(T)  # (N,m), (N,m,m) ascending
    lmin = eigvals[:, 0]
    lmax = eigvals[:, -1]
    y_min = eigvecs[:, :, 0]                  # (N, m)
    vmin = torch.einsum('Nmn,Nm->Nn', V, y_min)
    vmin = vmin / vmin.norm(dim=1, keepdim=True).clamp_min(eps)
    return lmin, lmax, vmin


def dense_eigs_reference(F, X):
    """Ground-truth (dense, O(n^2) autodiff Hessian) lambda_min/lambda_max
    for a small batch -- used ONLY in the diagnostic to validate Lanczos
    accuracy, never in the actual pipeline (too expensive at n=50)."""
    N, n = X.shape
    out_min, out_max = [], []
    for i in range(N):
        xi = X[i].detach().clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(lambda z: F(z[None]).sum(), xi)
        ev = torch.linalg.eigvalsh(H)
        out_min.append(ev[0].item())
        out_max.append(ev[-1].item())
    return torch.tensor(out_min), torch.tensor(out_max)


# ==== bundled from benchmarks_nd.py ====
"""
N-dimensional benchmark functions for the dimensionality sweep (Ext 2).

Ackley, Rastrigin, and Styblinski-Tang in see_core.py are already written
with .mean(1)/.sum(1) reductions over the last axis, which are the *same*
closed-form n-dimensional generalizations used throughout the optimization
literature (Ackley/Rastrigin/Styblinski-Tang all have standard n-dim forms
built exactly this way) -- so F_ackley/F_rastrigin/F_styblinski from
see_core.py are reused verbatim here, just called with X of shape (N, n)
for n != 2. Himmelblau and Levy are 2D-specific (no standard closed-form
n-dim analogue used in the literature) and are excluded from the
dimensionality sweep per the extension spec.
"""
FUNCS_ND = {'Ackley': F_ackley, 'Rastrigin': F_rastrigin, 'Styblinski': F_styblinski}
DOM_ND = {'Ackley': 5., 'Rastrigin': 5.12, 'Styblinski': 5.}  # same domain half-widths as the 2D baseline
DIMS = [2, 5, 10, 25, 50]


# ==== bundled from saddle_finder_nd.py ====
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


# ==== bundled from trajectory_nd.py ====
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


# ==== driver: ext2_dimsweep_rastrigin ====
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
