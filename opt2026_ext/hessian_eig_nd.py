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
