import torch.nn.functional as F
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self,w,q):
        """
        Initialize generator

        :param w: number of channels on the finest level
        :param q: latent space dimension
        """
        super(Generator, self).__init__()
        self.w = w
        self.fc = nn.Linear(q, w * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(w * 2, w, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(w, 1, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(w)
        self.bn2 = nn.BatchNorm2d(2*w)

    def forward(self, z):
        """
        :param z: latent space sample
        :return: g(z)
        """
        gz = self.fc(z)
        gz = gz.view(gz.size(0), self.w * 2, 7, 7)
        gz = self.bn2(gz)
        gz = F.relu(gz)
        gz = self.conv2(gz)
        gz = self.bn1(gz)

        gz = F.relu(gz)
        gz = torch.sigmoid(self.conv1(gz))
        return gz

class Encoder(nn.Module):
    def __init__(self,w,q):
        """
        Initialize the encoder for the VAE

        :param w: number of channels on finest level
        :param q: latent space dimension
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, w, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(w, w * 2, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(w * 2 * 7 * 7, q)
        self.fc_logvar = nn.Linear(w * 2 * 7 * 7, q)

    def forward(self, x):
        """
        :param x: MNIST image
        :return: mu,logvar that parameterize e(z|x) = N(mu, diag(exp(logvar)))
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Discriminator(nn.Module):
    def __init__(self, w,useSigmoid=True):
        """
        Discriminator for GANs
        :param w: number of channels on finest level
        :param useSigmoid: true --> DCGAN, false --> WGAN
        """
        super(Discriminator, self).__init__()
        self.w = w
        self.useSigmoid = useSigmoid
        self.conv1 = nn.Conv2d(1, w, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(w, w * 2, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(w * 2 * 7 * 7, 1)
        self.bn1 = nn.BatchNorm2d(w)
        self.bn2 = nn.BatchNorm2d(2*w)

    def forward(self,x):
        """
        :param x: MNIST image or generated image
        :return: d(x), value of discriminator
        """
        x = (x-0.5)/0.5 #
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,0.2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.leaky_relu(x,0.2)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        if self.useSigmoid:
            x = torch.sigmoid(x)
        return x
