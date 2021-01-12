import torch
from torch import nn
import numpy as np

import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE)

    """
    def __init__(self,e,g):
        """
        Initialize VAE
        :param e: encoder network e(z|x), provides parameters of approximate posteriors
        :param g: generator g(z), provides samples in data space
        """
        super(VAE, self).__init__()
        self.e = e
        self.g = g

    def ELBO(self,x):
        """
        Empirical lower bound on p_{\theta}(x)
        :param x: sample from dataset
        :return:
        """

        s = x.shape[0] # number of samples
        mu, logvar = self.e(x)  # parameters of approximate posterior
        z, eps = self.sample_ezx(x,mu, logvar)
        gz = self.g(z)

        log_pzx = torch.sum(self.log_prob_pzx(z,x,gz)[0])

        log_ezx = -0.5*torch.norm(eps)**2 - 0.5*torch.sum(logvar) - (z.shape[1]/2)*np.log(2*np.pi)

        return (-log_pzx+log_ezx)/s, (-log_pzx/s).item(), (log_ezx/s).item(), gz.detach(), mu.detach()

    def sample_ezx(self, x , mu=None, logvar=None, sample=None):
        """
        draw sample from approximate posterior

        :param x: sample from dataset
        :param mu: mean of approximate posterior (optional; will be computed here if is None)
        :param logvar: log-variance of approximate posterior (optional; will be computed here if is None)
        :param sample: flag whether to sample or return the mean
        :return:
        """
        if mu is None or logvar is None:
            mu, logvar = self.e(x)

        if sample is None:
            sample = self.training

        if sample:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return std * eps + mu, eps
        else:
            return mu, mu

    def log_prob_ezx(self,z,x):
        """
        :param z: latent sample
        :param x: data sample
        :return: log(e(z|x))
        """
        q = z.shape[1]

        mu, logvar = self.e(x)
        ezx = -torch.sum((0.5 / torch.exp(logvar)) * (z - mu) ** 2, dim=1) - 0.5 * torch.sum(logvar,dim=1) - (q/2)*np.log(2*np.pi)
        return ezx

    def log_prob_pzx(self,z,x,gz=None):
        """
        :param z: latent sample
        :param x: data sample
        :return: log(p(z,x)) = log(p(x|z)) + log(p(z)), log(p(x|z|), log(p(z))
        """
        if gz is None:
            gz = self.g(z)
        n = x.shape[1]
        px = -F.binary_cross_entropy(gz.view(-1, 784), x.view(-1, 784), reduction='none')
        px = torch.sum(px,dim=1)
        pz = - 0.5 * torch.norm(z, dim=1) ** 2  - (n/2)*np.log(2*np.pi)
        return px + pz, px, pz


