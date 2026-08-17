import torch

torch.set_default_dtype(torch.float64)

D_PARAM = 2 * 8 + 8 * 1 + 1

def xor_data(seed=42, n_per=50, noise=0.05):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y_base = torch.tensor([0., 1., 1., 0.])
    X = base.repeat_interleave(n_per, 0) + noise * torch.randn(4 * n_per, 2, generator=g)
    y = y_base.repeat_interleave(n_per, 0)
    return X, y

def unflatten(theta):
    W1 = theta[:16].reshape(2, 8)
    W2 = theta[16:24].reshape(8, 1)
    b2 = theta[24:25]
    return W1, W2, b2

def make_loss_fn(X, y):
    def loss_one(theta):
        W1, W2, b2 = unflatten(theta)
        h = torch.tanh(X @ W1)
        out = (h @ W2 + b2).squeeze(-1)
        return ((out - y) ** 2).mean()
    return torch.func.vmap(loss_one)
