import sys
sys.path.insert(0, '/Users/arnavdhiman/Projects/SEE-OPT2026-clone/opt2026_ext')
import torch, numpy as np
from scipy import optimize
torch.set_default_dtype(torch.float64)
import hessian_eig_nd

IN_DIM, HIDDEN, OUT_DIM = 2, 2, 1
N_PARAMS = IN_DIM*HIDDEN + HIDDEN + HIDDEN*OUT_DIM + OUT_DIM
_X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
_Y = torch.tensor([0.,1.,1.,0.])

def F(Theta):
    N = Theta.shape[0]; i=0
    W1 = Theta[:, i:i+IN_DIM*HIDDEN].reshape(N, IN_DIM, HIDDEN); i+=IN_DIM*HIDDEN
    b1 = Theta[:, i:i+HIDDEN]; i+=HIDDEN
    W2 = Theta[:, i:i+HIDDEN*OUT_DIM].reshape(N, HIDDEN, OUT_DIM); i+=HIDDEN*OUT_DIM
    b2 = Theta[:, i:i+OUT_DIM]; i+=OUT_DIM
    h = torch.tanh(torch.einsum('bi,nih->nbh', _X, W1) + b1.unsqueeze(1))
    out = torch.sigmoid(torch.einsum('nbh,nho->nbo', h, W2).squeeze(-1) + b2)
    eps=1e-7
    return -(_Y*torch.log(out+eps)+(1-_Y)*torch.log(1-out+eps)).mean(1)

def unpack(t):
    W1=t[:IN_DIM*HIDDEN].reshape(IN_DIM,HIDDEN); b1=t[IN_DIM*HIDDEN:IN_DIM*HIDDEN+HIDDEN]
    W2=t[IN_DIM*HIDDEN+HIDDEN:IN_DIM*HIDDEN+HIDDEN+HIDDEN]; b2=t[-1]
    return W1,b1,W2,b2
def pack(W1,b1,W2,b2): return np.concatenate([W1.flatten(),b1,W2,[b2]])

# tie the 2 units into 1 effective unit from the start, train within tied subspace
rng = np.random.default_rng(2)
w_shared = rng.normal(0,0.5,IN_DIM); b_shared = rng.normal(0,0.5); w2_tot = rng.normal(0,0.5)
theta0 = np.zeros(N_PARAMS)
theta0[:IN_DIM*HIDDEN] = np.repeat(w_shared, HIDDEN)
theta0[IN_DIM*HIDDEN:IN_DIM*HIDDEN+HIDDEN] = b_shared
theta0[IN_DIM*HIDDEN+HIDDEN:IN_DIM*HIDDEN+HIDDEN+HIDDEN] = w2_tot/HIDDEN
theta0[-1] = 0.0
print("N_PARAMS =", N_PARAMS, " initial loss:", F(torch.tensor(theta0)[None])[0].item())

theta = torch.tensor(theta0, requires_grad=True)
lr=0.5
for step in range(4000):
    loss = F(theta[None])[0]; loss.backward()
    with torch.no_grad(): theta -= lr*theta.grad
    theta.grad=None
print(f"tied-subspace training final loss: {loss.item():.6f}  (nonzero expected: 1 unit can't fit XOR)")

def gf(p):
    x=torch.tensor(p)[None].requires_grad_(True); F(x).sum().backward(); return x.grad[0].numpy()
sol,info,ier,msg = optimize.fsolve(gf, theta.detach().numpy(), full_output=True)
gn = np.linalg.norm(gf(sol))
print(f"fsolve refine: ier={ier} grad_norm={gn:.2e} max|sol|={np.abs(sol).max():.3f}")

X = torch.tensor(sol)[None]
lmin,lmax = hessian_eig_nd.batched_lanczos_extreme_eigs(F, X, m=N_PARAMS, device='cpu')
print(f"loss={F(X)[0].item():.6f}  lambda_min={lmin.item():.6f}  lambda_max={lmax.item():.6f}")
print(f"IS SADDLE? {lmin.item()<-1e-4 and lmax.item()>1e-4}")
