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
