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
