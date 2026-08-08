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

Numerical robustness (found via a real Kaggle failure, not hypothetical):
some trials in a batch diverge (e.g. Styblinski-Tang's x^4 term under a
large LR blows the trajectory up before nan_to_num clips the *position* to
1e6 -- but the Hessian-vector products computed FROM that huge-but-finite
position can still overflow to inf, or the Lanczos recurrence can degenerate
when a direction vector collapses). A single non-finite or ill-conditioned
row in the batched tridiagonal matrix makes torch.linalg.eigvalsh/eigh raise
for the WHOLE batch (LAPACK's batched routines fail all-or-nothing per call).
We therefore (a) sanitize every intermediate quantity in the Lanczos
recurrence, and (b) wrap the final eigendecomposition in a jitter-retry /
per-row fallback so one diverged trial can't crash the other 199 in the
batch. A row whose Hessian estimate is unrecoverable is treated as having
escaped (lambda_min set to a large positive sentinel) -- physically
reasonable, since a trajectory that diverged this badly has certainly left
the saddle's local basin, matching what criterion A (distance) would already
say about the same point.
"""
import torch

_CLAMP = 1e8      # intermediate magnitude clamp -- generous vs. saddle-local scales, well inside float64 range
_ESCAPED_SENTINEL = 1e6  # lambda_min/lambda_max value assigned to unrecoverable (diverged) rows


def _sanitize(t, clamp=_CLAMP):
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)
    return t.clamp(-clamp, clamp)


def batched_hvp(F, X, V):
    """X: (N,n) requires no pre-existing grad state (we set it up here).
    V: (N,n) per-row direction vectors.
    Returns (N,n): Hv[i] = Hessian(f)(X[i]) @ V[i]. Exact (autodiff), not
    finite-difference -- no h-parameter to tune. Sanitized against overflow
    from diverged trajectories (see module docstring)."""
    Xc = X.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(F(Xc).sum(), Xc, create_graph=True)[0]  # (N,n), per-row correct since F is row-separable
    g = _sanitize(g)
    gv = (g * V).sum()
    Hv = torch.autograd.grad(gv, Xc, retain_graph=False)[0]  # (N,n)
    return _sanitize(Hv.detach())


def _safe_tridiag_eigh(T, want_vectors=False):
    """torch.linalg.eigvalsh/eigh on a batched tridiagonal matrix, robust to
    the rare diverged-trial row: sanitize -> try -> jitter+retry -> per-row
    numpy fallback with an escaped-sentinel for any row still unrecoverable.
    Returns (eigvals, eigvecs_or_None)."""
    N = T.shape[0]
    T = _sanitize(T)
    fn = torch.linalg.eigh if want_vectors else torch.linalg.eigvalsh
    try:
        return fn(T) if want_vectors else (fn(T), None)
    except Exception:
        pass
    # jitter retry (fixes most "ill-conditioned / repeated eigenvalues" LAPACK failures)
    eye = torch.eye(T.shape[-1], device=T.device, dtype=T.dtype).unsqueeze(0)
    try:
        Tj = T + 1e-6 * eye
        return fn(Tj) if want_vectors else (fn(Tj), None)
    except Exception:
        pass
    # per-row fallback: isolate the bad row(s), sentinel them, solve the rest individually
    import numpy as np
    m = T.shape[-1]
    eigvals = torch.full((N, m), _ESCAPED_SENTINEL, device=T.device, dtype=T.dtype)
    eigvecs = torch.zeros((N, m, m), device=T.device, dtype=T.dtype) if want_vectors else None
    if want_vectors:
        eigvecs[:, 0, :] = 1.0  # trivial fallback eigenvector (e_0) for unrecoverable rows
    Tcpu = T.detach().cpu().numpy()
    for i in range(N):
        try:
            if want_vectors:
                ev, evec = np.linalg.eigh(Tcpu[i])
                eigvecs[i] = torch.tensor(evec, device=T.device, dtype=T.dtype)
            else:
                ev = np.linalg.eigvalsh(Tcpu[i])
            eigvals[i] = torch.tensor(ev, device=T.device, dtype=T.dtype)
        except Exception:
            pass  # leave this row at the escaped sentinel
    return eigvals, eigvecs


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
        alpha = _sanitize((w * v).sum(dim=1))
        alphas.append(alpha)
        w = _sanitize(w - alpha.unsqueeze(1) * v - beta_prev.unsqueeze(1) * v_prev)
        beta = _sanitize(w.norm(dim=1))
        betas.append(beta)
        v_prev = v
        beta_safe = beta.clamp_min(eps)
        v = w / beta_safe.unsqueeze(1)
        v = _sanitize(v, clamp=1e3)  # a unit-ish vector should never need a huge clamp
        beta_prev = beta

    A = torch.stack(alphas, dim=1)          # (N, m)
    B = torch.stack(betas[:-1], dim=1) if m > 1 else torch.zeros(N, 0, device=device, dtype=X.dtype)  # (N, m-1)

    T = torch.diag_embed(A)                  # (N, m, m)
    if m > 1:
        idx = torch.arange(m - 1, device=device)
        T[:, idx, idx + 1] = B
        T[:, idx + 1, idx] = B

    eigvals, _ = _safe_tridiag_eigh(T, want_vectors=False)
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
        alpha = _sanitize((w * v).sum(dim=1))
        alphas.append(alpha)
        w = _sanitize(w - alpha.unsqueeze(1) * v - beta_prev.unsqueeze(1) * v_prev)
        beta = _sanitize(w.norm(dim=1))
        betas.append(beta)
        v_prev = v
        beta_safe = beta.clamp_min(eps)
        v = w / beta_safe.unsqueeze(1)
        v = _sanitize(v, clamp=1e3)
        beta_prev = beta

    A = torch.stack(alphas, dim=1)
    B = torch.stack(betas[:-1], dim=1) if m > 1 else torch.zeros(N, 0, device=device, dtype=X.dtype)
    V = torch.stack(V_stack, dim=1)  # (N, m, n)

    T = torch.diag_embed(A)
    if m > 1:
        idx = torch.arange(m - 1, device=device)
        T[:, idx, idx + 1] = B
        T[:, idx + 1, idx] = B

    eigvals, eigvecs = _safe_tridiag_eigh(T, want_vectors=True)
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


# ==== bundled from xor_network.py ====
"""
The first real (non-benchmark-function) loss landscape tested: a tiny MLP
trained on XOR, treated as a function of its flattened parameter vector so
the existing nD saddle-finding/criterion-evaluation pipeline applies
unchanged -- theta plays exactly the role x played for the benchmark
functions.

Architecture: 2 (input) -> 2 (hidden, tanh) -> 1 (output, sigmoid), BCE loss.
Parameter count: W1 (2x2=4) + b1 (2) + W2 (2x1=2) + b2 (1) = 9.

SADDLE-FINDING METHOD -- this required real debugging, documented here so
the choice isn't a black box:

Global random multi-start fsolve (the method that works for the benchmark
functions) FAILS on this landscape. Diagnosed empirically before committing
to any Kaggle run: tanh/sigmoid saturate at large weight magnitudes, so
||grad|| -> 0 out at |theta| -> infinity too, not just at genuine interior
critical points. fsolve's Newton iteration overwhelmingly "converges" by
running off to |theta| in the thousands-to-millions where gradients vanish
trivially through saturation -- not a real critical point. Across 300 random
starts in [-3,3]^9-17, ZERO landed on an in-domain, non-saturated critical
point (see opt2026_ext/verification/xor_saddle_search_diagnosis.py).

The fix: construct the saddle directly from the network's known symmetry,
following the standard "tie units together" mechanism for saddle points in
small neural nets (e.g. Dauphin et al. 2014). With H=2 hidden tanh units,
tying them into a single EFFECTIVE unit (identical incoming weights, bias,
and outgoing weight) is provably insufficient to represent XOR -- a single
tanh unit's output is monotonic in one pre-activation, so sigmoid(w2*tanh(.)+b2)
can only express a decision rule based on one hyperplane, and XOR is not
separable by any one hyperplane. Training WITHIN this tied (rank-collapsed)
subspace via deterministic gradient descent (mirrored updates keep the tie
exact -- no noise to break the symmetry) converges to a critical point of
the constrained problem. Verified this is ALSO a genuine saddle of the FULL
(untied) 9-parameter space: lambda_min=-0.0152, lambda_max=+0.462 (both
comfortably clear the same +-1e-4 acceptance thresholds used everywhere
else in this project), fsolve-refined to grad_norm~2e-17, loss=ln(2)=0.693147
exactly matches the closed-form value for a network collapsed to constant
output 0.5 -- untying the pair strictly helps (negative curvature), some
other direction strictly hurts (positive curvature): a textbook saddle.

Earlier attempts documented for completeness (also in verification/):
attempting the same construction with H=4 hidden units, tying either all 4
or just 2 of them, both converged to points where the network fit XOR
near-perfectly (loss ~1e-3 to 1e-16) -- because 2-3 effective units ARE
enough capacity to solve XOR, so those constructions found genuine minima,
not saddles. Near those near-perfect-fit points the sigmoid/tanh saturate
so completely that the ENTIRE Hessian goes flat (lambda_min=lambda_max=0.0
to printed precision) -- informative in its own right (this project's fixed
+-1e-4 curvature thresholds, calibrated on benchmark functions, don't
transfer to near-saturated neural network regimes without care) but not
useful as a saddle for this experiment. H=2 avoids this because the tied
point's loss floor (ln 2) is bounded well away from zero, so saturation
never fully sets in.
"""
import torch
import numpy as np

IN_DIM = 2
HIDDEN = 2
OUT_DIM = 1
N_PARAMS = IN_DIM * HIDDEN + HIDDEN + HIDDEN * OUT_DIM + OUT_DIM  # 4+2+2+1 = 9

_X_XOR = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
_Y_XOR = torch.tensor([0., 1., 1., 0.])


def F_xor_loss(Theta):
    """Theta: (N, 9) batched flattened parameter vectors.
    Returns (N,): mean BCE loss over the 4 XOR points, per parameter vector.
    Row-separable, compatible with bgrad/batched_hvp's sum-then-backward
    trick exactly like the benchmark functions."""
    N = Theta.shape[0]
    i = 0
    W1 = Theta[:, i:i + IN_DIM * HIDDEN].reshape(N, IN_DIM, HIDDEN); i += IN_DIM * HIDDEN
    b1 = Theta[:, i:i + HIDDEN]; i += HIDDEN
    W2 = Theta[:, i:i + HIDDEN * OUT_DIM].reshape(N, HIDDEN, OUT_DIM); i += HIDDEN * OUT_DIM
    b2 = Theta[:, i:i + OUT_DIM]; i += OUT_DIM

    Xd = _X_XOR.to(Theta.device, Theta.dtype)
    yd = _Y_XOR.to(Theta.device, Theta.dtype)

    h = torch.tanh(torch.einsum('bi,nih->nbh', Xd, W1) + b1.unsqueeze(1))
    out = torch.sigmoid(torch.einsum('nbh,nho->nbo', h, W2).squeeze(-1) + b2)
    eps = 1e-7
    bce = -(yd * torch.log(out + eps) + (1 - yd) * torch.log(1 - out + eps)).mean(1)
    return bce


def _unpack(t):
    W1 = t[:IN_DIM * HIDDEN].reshape(IN_DIM, HIDDEN)
    b1 = t[IN_DIM * HIDDEN:IN_DIM * HIDDEN + HIDDEN]
    W2 = t[IN_DIM * HIDDEN + HIDDEN:IN_DIM * HIDDEN + HIDDEN + HIDDEN]
    b2 = t[-1]
    return W1, b1, W2, b2


def _pack(W1, b1, W2, b2):
    return np.concatenate([W1.flatten(), b1, W2, [b2]])


def find_xor_saddle(seed=0, lr=0.5, steps=4000, device='cpu'):
    """Constructive saddle-finder (see module docstring for why global
    multi-start fsolve doesn't work here). Returns a verified torch tensor
    (N_PARAMS,) saddle point, plus a diagnostics dict."""
    from scipy import optimize
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(seed)
    w_shared = rng.normal(0, 0.5, IN_DIM)
    b_shared = rng.normal(0, 0.5)
    w2_tot = rng.normal(0, 0.5)
    theta0 = np.zeros(N_PARAMS)
    theta0[:IN_DIM * HIDDEN] = np.repeat(w_shared, HIDDEN)  # tie: every column identical
    theta0[IN_DIM * HIDDEN:IN_DIM * HIDDEN + HIDDEN] = b_shared
    theta0[IN_DIM * HIDDEN + HIDDEN:IN_DIM * HIDDEN + HIDDEN + HIDDEN] = w2_tot / HIDDEN
    theta0[-1] = 0.0

    theta = torch.tensor(theta0, requires_grad=True)
    for _ in range(steps):
        loss = F_xor_loss(theta[None])[0]
        loss.backward()
        with torch.no_grad():
            theta -= lr * theta.grad
        theta.grad = None
    tied_loss = loss.item()

    def gf(p):
        x = torch.tensor(p)[None].requires_grad_(True)
        F_xor_loss(x).sum().backward()
        return x.grad[0].numpy()

    sol, info, ier, msg = optimize.fsolve(gf, theta.detach().numpy(), full_output=True)
    grad_norm = float(np.linalg.norm(gf(sol)))

    s = torch.tensor(sol, device=device)
    diag = {'tied_subspace_loss': tied_loss, 'fsolve_ier': ier, 'fsolve_msg': msg,
            'grad_norm': grad_norm, 'max_abs_param': float(np.abs(sol).max())}
    return s, diag


DOM_XOR = 3.0  # kept for interface parity with the benchmark-function saddle finders; not used by find_xor_saddle


# ==== driver: ext4_nn_saddle ====
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
        print(f"[ext4] CUDA present but not usable ({ex}); falling back to CPU.")
        return torch.device('cpu')

device = pick_device()
torch.set_default_dtype(torch.float64)
print(f"[ext4] device={device}  torch={torch.__version__}  numpy={np.__version__}  scipy={__import__('scipy').__version__}")

OUTDIR = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
RAWDIR = os.path.join(OUTDIR, 'raw')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAWDIR, exist_ok=True)
rng = np.random.default_rng(SEED)

print(f'\n=== Ext4: first real (non-benchmark) loss landscape -- XOR MLP, {N_PARAMS}-dim parameter space ===')
print('Saddle constructed via tied-unit symmetry (see xor_network.py docstring for why global')
print('multi-start fsolve fails on this saturating landscape).')

# ---------------------------------------------------------------- 1. saddle: CPU-verified fixed coordinates
# The saddle construction (find_xor_saddle) is a fragile 4000-step iterative
# tied-subspace gradient descent -- of 6 random seeds tried locally, only
# seed=2 converges to the clean saddle (loss=ln(2), lambda_min=-0.0152); the
# other 5 land in a degenerate attractor (lambda_min=lambda_max~0, huge
# param magnitude). A first Kaggle GPU run (kernel version 1) reproduced
# that SAME degenerate failure even with seed=2 -- GPU/CPU float64
# non-determinism compounded over 4000 steps landed in a different basin
# than the CPU run did (this is exactly the class of non-determinism the
# baseline README already documents as a known limitation). Rather than gamble
# on construction being GPU-reproducible, the fragile part (construction) is
# done once on CPU and hardcoded here; only the trial-grid forward simulation
# (which every other kernel in this project already does successfully on
# GPU from a fixed starting point) runs on GPU.
_THETA_SADDLE = [-1.0894063429134349e-15, -1.0894063429134349e-15, -1.0894063429134349e-15,
                 -1.0894063429134349e-15, -0.11738607773232793, -0.11738607773232793,
                 -0.5271470428799861, -0.5271470428799861, -0.12319411588445499]
s = torch.tensor(_THETA_SADDLE, device=device)
lmin_t, lmax_t, vmin_t = batched_lanczos_min_eigpair(F_xor_loss, s[None], m=N_PARAMS, device=device)
lmin_v, lmax_v = lmin_t.item(), lmax_t.item()
r_curv = 1 / np.sqrt(abs(lmin_v))
f_s = F_xor_loss(s[None])[0].item()
geom = (s, vmin_t[0], r_curv, f_s, lmin_v)

is_saddle = lmin_v < -1e-4 and lmax_v > 1e-4
print(f'\nFixed (CPU-verified) saddle recomputed on {device}: loss={f_s:.6f}  lambda_min={lmin_v:.6f}  '
      f'lambda_max={lmax_v:.6f}  r_curv={r_curv:.4f}')
print(f'  ACCEPTED AS SADDLE (lmin<-1e-4 and lmax>1e-4): {is_saddle}')
print(f'  (CPU reference values: loss=0.693147 lambda_min=-0.015189 lambda_max=0.462421 -- '
      f'comparing to catch any residual device-dependent eigenvalue drift)')
if not is_saddle:
    raise RuntimeError('Fixed saddle point failed acceptance on this device -- do not proceed with trial grid.')

pd.DataFrame([{'loss': f_s, 'lambda_min': lmin_v, 'lambda_max': lmax_v, 'r_curv': r_curv,
               'n_params': N_PARAMS, 'device': str(device)}]
             ).to_csv(os.path.join(OUTDIR, 'ext4_xor_saddle_report.csv'), index=False)

# ---------------------------------------------------------------- 2. trial grid: all 7 optimizers x 6 LRs x 4 criteria
print(f'\nRunning trial grid: {len(OPTS_EXT)} optimizers x {len(LRS)} LRs = {len(OPTS_EXT)*len(LRS)} configs, '
      f'N={N} trials, T={TMAX} steps')
DATA = {}
t_start = time.time()
for o in OPTS_EXT:
    for base_lr in LRS:
        actual_lr = lr_for(o, base_lr)
        e, stp = run_config_nd(F_xor_loss, geom, o, actual_lr, make_opt, device=device, N=N, TMAX=TMAX,
                                seed=SEED, lanczos_m=N_PARAMS, VAR=VAR)
        DATA[(o, base_lr)] = (e, stp, actual_lr)
    print(f'  {o} done. ({time.time()-t_start:.1f}s elapsed)')

with open(os.path.join(RAWDIR, 'ext4_xor_raw_trials.pkl'), 'wb') as f:
    pickle.dump({'DATA': DATA, 'geom': (s.cpu().numpy(), vmin_t[0].cpu().numpy(), r_curv, f_s, lmin_v),
                 'VAR': VAR, 'OPTS_EXT': OPTS_EXT, 'LRS': LRS, 'SEED': SEED, 'N': N, 'TMAX': TMAX,
                 'N_PARAMS': N_PARAMS, 'theta_saddle': _THETA_SADDLE}, f)
print(f'Raw trial data saved to {RAWDIR}/ext4_xor_raw_trials.pkl')

# ---------------------------------------------------------------- 3. SEE tables
best_rows = []
for o in OPTS_EXT:
    vals = []
    for fam in FAMS:
        fi = headline_idx(fam)
        v_best = max(see_pt(DATA[(o, lr)][0][fi], DATA[(o, lr)][1][fi]) for lr in LRS)
        vals.append(v_best)
    best_rows.append({'optimizer': o, **{f'best_{f}': v for f, v in zip(FAMS, vals)}})
    print(f'  {o:13s}: ' + ' / '.join(f'{v:.3f}' for v in vals))
df_best = pd.DataFrame(best_rows)
df_best.to_csv(os.path.join(OUTDIR, 'ext4_xor_best_lr.csv'), index=False)

# ---------------------------------------------------------------- 4. the key question: does B-D still agree, does A-B still disagree?
print('\n=== The key question: does the criterion-dependence pattern replicate on a real network? ===')
spearman_rows = []
for f1, f2 in [('A', 'B'), ('A', 'D'), ('B', 'D'), ('A', 'C'), ('B', 'C'), ('C', 'D')]:
    a = df_best[f'best_{f1}'].values
    b = df_best[f'best_{f2}'].values
    rho, n_used = spearman_with_n(a, b)
    spearman_rows.append({'pair': f'{f1}-{f2}', 'rho': rho, 'n': n_used})
    print(f'  {f1}-{f2}: rho={rho:+.3f}  (n={n_used})')
pd.DataFrame(spearman_rows).to_csv(os.path.join(OUTDIR, 'ext4_xor_spearman.csv'), index=False)

V = np.array([[r[f'best_{f}'] for r in best_rows] for f in FAMS])
w_res = kendalls_w_ci(V, rng=rng, method='bootstrap')
print(f"\n  Kendall's W = {w_res['W']:.3f}  95% CI [{w_res['ci_lo']:.3f}, {w_res['ci_hi']:.3f}]  (n_items={w_res['n_items']})")
pd.DataFrame([{'W': w_res['W'], 'n_items': w_res['n_items'], 'n_judges': w_res['n_judges'],
               'ci_lo': w_res['ci_lo'], 'ci_hi': w_res['ci_hi'], 'method': w_res['method'], 'B': w_res['B']}]
             ).to_csv(os.path.join(OUTDIR, 'ext4_xor_kendall_w.csv'), index=False)

print(f'\nExt4 done. Total elapsed: {time.time()-t_start:.1f}s. Files in {OUTDIR}/')
