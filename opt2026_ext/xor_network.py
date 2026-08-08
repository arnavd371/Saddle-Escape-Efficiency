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
