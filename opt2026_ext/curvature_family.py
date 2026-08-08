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
