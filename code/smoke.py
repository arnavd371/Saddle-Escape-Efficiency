import time
import numpy as np
import torch
import core
import algorithm as see

t0 = time.time()
for name in core.FUNCS2D:
    F, L = core.FUNCS2D[name]
    ss = core.find_saddles_2d(F, L)
    print(name, len(ss), [np.round(s, 3) for s in ss])
print('2D saddle search', round(time.time() - t0, 1))

F, L = core.FUNCS2D['Levy']
ss = core.find_saddles_2d(F, L)
s = torch.tensor(ss[0])[None]
lmn, lmx, v = core.eigh_min(F, s)
lmn2, lmx2 = core.lanczos_extremes(F, s, k=20)
print('levy eigh', lmn.item(), lmx.item())
print('levy lanczos', lmn2[0], lmx2[0])

s0 = torch.tensor(ss[0])
_, _, v0 = core.eigh_min(F, s0[None])
v0 = v0[0] / v0[0].norm()
r_curv = 1 / np.sqrt(abs(lmn.item()))
f_s = F(s0[None])[0].item()

def lambda_fn(Xd):
    lmin, _ = core.lanczos_extremes(F, Xd, k=20)
    return lmin

esc, stp = see.simulate(
    F, s0, v0, r_curv, f_s, 'Adam', 0.1,
    N=20, Tmax=30, seed=42, families=['A', 'B', 'C', 'D'], lambda_fn=lambda_fn)
k = see.headline_idx('A')
print('esc frac A', esc[k].mean())
pt, lo, hi = see.see_ci(esc[k], stp[k], np.random.default_rng(0))
print('SEE_A', pt, lo, hi)
print('total', round(time.time() - t0, 1))
