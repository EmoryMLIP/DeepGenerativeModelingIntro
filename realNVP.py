import torch
from torch import nn

class NF(nn.Module):
    """
    Normalizing flow for density estimation and sampling

    """
    def __init__(self, layers, prior):
        """
        Initialize normalizing flow

        :param layers: list of layers f_j with tractable inverse and log-determinant (e.g., RealNVPLayer)
        :param prior: latent distribution, e.g., distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
        """
        super(NF, self).__init__()
        self.prior = prior
        self.layers = layers

    def g(self, z):
        """
        :param z: latent variable
        :return: g(z) and hidden states
        """
        y = z
        ys = [torch.clone(y).detach()]
        for i in range(len(self.layers)):
            y, _ = self.layers[i].f(y)
            ys.append(torch.clone(y).detach())

        return y, ys

    def ginv(self, x):
        """
        :param x: sample from dataset
        :return: g^(-1)(x), value of log-determinant, and hidden layers
        """
        p = x
        log_det_ginv = torch.zeros(x.shape[0])
        ps = [torch.clone(p).detach()]
        for i in reversed(range(len(self.layers))):
            p, log_det_finv = self.layers[i].finv(p)
            ps.append(torch.clone(p).detach().cpu())
            log_det_ginv += log_det_finv

        return p, log_det_ginv, ps

    def log_prob(self, x):
        """
        Compute log-probability of a sample using change of variable formula

        :param x: sample from dataset
        :return: logp_{\theta}(x)
        """
        z, log_det_ginv, _ = self.ginv(x)
        return self.prior.log_prob(z) + log_det_ginv

    def sample(self, s):
        """
        Draw random samples from p_{\theta}

        :param s: number of samples to draw
        :return:
        """
        z = self.prior.sample((s, 1)).squeeze(1)
        x, _ = self.g(z)
        return x


class RealNVPLayer(nn.Module):
    """
    Real non-volume preserving flow layer

    Reference: Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016, May 27).
               Density estimation using Real NVP. arXiv.org.
    """
    def __init__(self, s, t, mask):
        """
        Initialize real NVP layer
        :param s: network to compute the shift
        :param t: network to compute the translation
        :param mask: splits the feature vector into two parts
        """
        super(RealNVPLayer, self).__init__()
        self.mask = mask
        self.t = t
        self.s = s

    def f(self, y):
        """
        apply the layer function f
        :param y:  feature vector
        :return:
        """
        y1 = y * self.mask
        s = self.s(y1)
        t = self.t(y1)
        y2 = (y * torch.exp(s) + t) * (1 - self.mask)
        return y1 + y2, torch.sum(s, dim=1)

    def finv(self, y):
        """
        apply the inverse of the layer function
        :param y: feature vector
        :return:
        """
        y1 = self.mask * y
        s = self.s(y1)
        t = self.t(y1)
        y2 = (1 - self.mask) * (y - t) * torch.exp(-s)
        return y1 + y2, -torch.sum(s, dim=1)


if __name__ == "__main__":
    # layers and masks
    K = 6
    w = 128
    layers = torch.nn.ModuleList()
    for k in range(K):
        mask = torch.tensor([1 - (k % 2), k % 2])
        t = nn.Sequential(nn.Linear(2, w), nn.LeakyReLU(), nn.Linear(w, w), nn.LeakyReLU(), nn.Linear(w, 2),
                          nn.Tanh())
        s = nn.Sequential(nn.Linear(2, w), nn.LeakyReLU(), nn.Linear(w, w), nn.LeakyReLU(), nn.Linear(w, 2),
                          nn.Tanh())
        layer = RealNVPLayer(s, t, mask)
        layers.append(layer)

    from torch import distributions
    prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    flow = NF(layers, prior)

    x = flow.sample(200).detach()
    # test inverse
    xt = flow.ginv(flow.g(x)[0])[0].detach()
    print(torch.norm(x - xt) / torch.norm(x))

    # test inverse
    xt = flow.g(flow.ginv(x)[0])[0].detach()
    print(torch.norm(x - xt) / torch.norm(x))


