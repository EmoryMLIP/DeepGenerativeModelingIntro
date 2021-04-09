import torch
from torch import nn
from torch.nn.functional import pad
from Phi import *
from torch import distributions


class OTFlow(nn.Module):
    """
    OT-Flow for density estimation and sampling as described in

    @article{onken2020otflow,
        title={OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport},
        author={Derek Onken and Samy Wu Fung and Xingjian Li and Lars Ruthotto},
        year={2020},
        journal = {arXiv preprint arXiv:2006.00104},
    }

    """
    def __init__(self, net, nt, alph, prior, T=1.0):
        """
        Initialize OT-Flow

        :param net: network for value function
        :param nt: number of rk4 steps
        :param alph: penalty parameters
        :param prior: latent distribution, e.g., distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
        """
        super(OTFlow, self).__init__()
        self.prior = prior
        self.nt = nt
        self.T = T
        self.net = net
        self.alph = alph


    def g(self, z, nt = None, storeAll=False):
        """
        :param z: latent variable
        :return: g(z) and hidden states
        """
        return self.integrate(z,[self.T, 0.0], nt,storeAll)

    def ginv(self, x, nt=None, storeAll=False):
        """
        :param x: sample from dataset
        :return: g^(-1)(x), value of log-determinant, and hidden layers
        """

        return self.integrate(x,[0.0, self.T], nt,storeAll)

    def log_prob(self, x, nt=None):
        """
        Compute log-probability of a sample using change of variable formula

        :param x: sample from dataset
        :return: logp_{\theta}(x)
        """
        z, _, log_det_ginv, v, r = self.ginv(x,nt)
        return self.prior.log_prob(z) + log_det_ginv, v, r

    def sample(self, s,nt=None):
        """
        Draw random samples from p_{\theta}

        :param s: number of samples to draw
        :return:
        """
        z = self.prior.sample((s, 1)).squeeze(1)
        x, _, _, _, _ = self.g(z,nt)
        return x

    def f(self,x, t):
        """
        neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
        the HJB regularizer (see Eq. (7)).

        d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

        x - particle position
        l - log determinant
        v - accumulated transport costs (Lagrangian)
        r - accumulates violation of HJB condition along trajectory
        """
        nex, d = x.shape
        z = pad(x[:, :d], (0, 1, 0, 0), value=t)
        gradPhi, trH = self.net.trHess(z)

        dx = -(1.0 / self.alph[0]) * gradPhi[:, 0:d]
        dl = -(1.0 / self.alph[0]) * trH
        dv = 0.5 * torch.sum(torch.pow(dx, 2), 1)
        dr = torch.abs(-gradPhi[:, -1] + self.alph[0] * dv)

        return dx, dl, dv, dr

    def integrate(self, y, tspan, nt=None,storeAll=False):
        """
        RK4 time-stepping to integrate the neural ODE

        :param y: initial state
        :param tspan: time interval (can go backward in time)
        :param nt: number of time steps (default is self.nt)
        :return: y (final state), ys (all states), l (log determinant), v (transport costs), r (HJB penalty)
        """
        if nt is None:
            nt = self.nt

        nex, d = y.shape
        h = (tspan[1] - tspan[0])/ nt
        tk = tspan[0]

        l = torch.zeros((nex), device=y.device, dtype=y.dtype)
        v = torch.zeros((nex), device=y.device, dtype=y.dtype)
        r = torch.zeros((nex), device=y.device, dtype=y.dtype)
        if storeAll:
            ys = [torch.clone(y).detach().cpu()]
        else:
            ys = None

        w =  [(h/6.0),2.0*(h/6.0),2.0*(h/6.0),1.0*(h/6.0)]
        for i in range(nt):
            y0 = y

            dy, dl, dv, dr = self.f(y0, tk)
            y = y0 + w[0] * dy
            l += w[0] * dl
            v += w[0] * dv
            r += w[0] * dr

            dy, dl, dv, dr =  self.f(y0 + 0.5 * h * dy, tk + (h / 2))
            y += w[1] * dy
            l += w[1] * dl
            v += w[1] * dv
            r += w[1] * dr

            dy, dl, dv, dr = self.f(y0 + 0.5 * h * dy, tk + (h / 2))
            y += w[2] * dy
            l += w[2] * dl
            v += w[2] * dv
            r += w[2] * dr

            dy, dl, dv, dr = self.f(y0 + h * dy, tk + h)
            y += w[3] * dy
            l += w[3] * dl
            v += w[3] * dv
            r += w[3] * dr

            if storeAll:
                ys.append(torch.clone(y).detach().cpu())
            tk +=h

        return y, ys, l, v, r

if __name__ == "__main__":
    # layers and masks

    nt =  16
    alph = [1.0, 5.0, 10.0]
    prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    net = Phi(nTh=2, m=16, d=2, alph=alph)
    T=1.0

    flow = OTFlow(net,nt,alph,prior,T)

    x = flow.sample(200).detach()
    # test inverse
    xt = flow.ginv(flow.g(x)[0])[0].detach()
    print(torch.norm(x - xt) / torch.norm(x))

    # test inverse
    xt = flow.g(flow.ginv(x)[0])[0].detach()
    print(torch.norm(x - xt) / torch.norm(x))


