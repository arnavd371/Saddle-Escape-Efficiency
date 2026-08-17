import sys
sys.path.insert(0, '/Users/arnavdhiman/Projects/SEE-OPT2026-clone/opt2026_ext')
import torch, numpy as np
from scipy import optimize
torch.set_default_dtype(torch.float64)
from xor_network import F_xor_loss, N_PARAMS, DOM_XOR

def gf(p):
    x = torch.tensor(p)[None].requires_grad_(True)
    F_xor_loss(x).sum().backward()
    return x.grad[0].numpy()

rng = np.random.default_rng(0)
ier_counts = {}
gnorms = []
max_abs = []
for i in range(50):
    p0 = rng.uniform(-DOM_XOR, DOM_XOR, N_PARAMS)
    sol, info, ier, msg = optimize.fsolve(gf, p0, full_output=True)
    ier_counts[ier] = ier_counts.get(ier, 0) + 1
    gn = np.linalg.norm(gf(sol))
    gnorms.append(gn)
    max_abs.append(np.abs(sol).max())
    if i < 8:
        print(f"start {i}: ier={ier} msg={msg[:60]!r} final_grad_norm={gn:.2e} max|sol|={np.abs(sol).max():.2f} loss0={F_xor_loss(torch.tensor(p0)[None])[0].item():.4f}")

print()
print("ier distribution:", ier_counts)
print(f"grad norm: min={min(gnorms):.2e} median={np.median(gnorms):.2e} max={max(gnorms):.2e}")
print(f"max|sol| in-domain(<=3.0): {sum(1 for m in max_abs if m<=DOM_XOR)}/{len(max_abs)}")
