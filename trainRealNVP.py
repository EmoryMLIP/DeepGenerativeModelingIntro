import torch
from torch import nn
import argparse
import numpy as np
from torch import distributions
from sklearn import datasets
import matplotlib.pyplot as plt

device = "cpu"

parser = argparse.ArgumentParser('RealNVP')
parser.add_argument("--batch_size"    , type=int, default=256, help="batch size")
parser.add_argument("--noise"    , type=int, default=0.05, help="noise of moons")
parser.add_argument("--width"    , type=int, default=128, help="width neural nets")
parser.add_argument("--K"    , type=int, default=6, help="number of layers")
parser.add_argument("--num_steps"    , type=int, default=20000, help="number of training steps")
parser.add_argument("--plot_interval"    , type=int, default=1000, help="plot solution every so many steps")
parser.add_argument("--out_file", type=str, default=None, help="base filename saving trained model (extension .pt), history (extension .mat), and intermediate plots (extension .png")

args = parser.parse_args()

from realNVP import NF, RealNVPLayer
K = args.K
w = args.width

layers = torch.nn.ModuleList()
for k in range(K):
    mask = torch.tensor([1 - (k % 2), k % 2])
    t = nn.Sequential(nn.Linear(2, w), nn.LeakyReLU(), nn.Linear(w, w), nn.LeakyReLU(), nn.Linear(w, 2),
                      nn.Tanh())
    s = nn.Sequential(nn.Linear(2, w), nn.LeakyReLU(), nn.Linear(w, w), nn.LeakyReLU(), nn.Linear(w, 2),
                      nn.Tanh())
    layer = RealNVPLayer(s, t, mask)
    layers.append(layer)

prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
flow = NF(layers, prior).to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)

his = np.zeros((0,1))

print((3*"--" + "device=%s, K=%d, width=%d, batch_size=%d, num_steps=%d" + 3*"--") % (device, args.K, args.width, args.batch_size, args.num_steps, ))

if args.out_file is not None:
    import os
    out_dir, fname = os.path.split(args.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print((3*"--" + "out_file: %s" + 3*"--") % (args.out_file))

print((2*"%7s    ") % ("step","J_ML"))


train_JML = 0.0
num_step = 0

for step in range(args.num_steps):

    x = torch.tensor(datasets.make_moons(n_samples=args.batch_size, noise=args.noise)[0], dtype=torch.float32)
    loss = -flow.log_prob(x).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_JML += loss.item()
    num_step += 1

    if (step + 1) % args.plot_interval == 0:
        train_JML /= num_step

        print(("%06d   " + "%1.4e  ") %
              (step + 1, train_JML))
        his = np.vstack([his, [train_JML]])

        zs = flow.ginv(x)[0].detach()
        xs = flow.sample(200).detach()
        x1 = torch.linspace(-1.2, 2.1, 100)
        x2 = torch.linspace(-1.2, 2.1, 100)
        xg = torch.meshgrid(x1, x2)
        xx = torch.cat((xg[0].reshape(-1, 1), xg[1].reshape(-1, 1)), 1)
        log_px = flow.log_prob(xx).detach()
        train_JML = 0.0
        num_step = 0

        plt.Figure()
        plt.rcParams.update({'font.size': 16, "text.usetex": True})

        plt.subplot(1,3,1)
        plt.plot(xs[:, 0], xs[:, 1], "bs")
        plt.axis((-1.2, 2.1, -1.2, 2.1))
        plt.xticks((-1.2, 2.1))
        plt.yticks((-1.2, 2.1))
        plt.xlabel("$\mathbf{x}_1$", labelpad=-20)
        plt.ylabel("$\mathbf{x}_2$", labelpad=-30)
        plt.title("$g_{\\theta}(\mathcal{Z})$")

        plt.subplot(1,3,2)
        plt.plot(zs[:, 0], zs[:, 1], "or")
        plt.axis((-3.5, 3.5, -3.5, 3.5))
        plt.xticks((-3.5, 3.5))
        plt.yticks((-3.5, 3.5))
        plt.xlabel("$\mathbf{z}_1$", labelpad=-20)
        plt.ylabel("$\mathbf{z}_2$", labelpad=-30)
        plt.title("$g^{-1}_{\\theta}(\mathcal{Z})$")

        plt.subplot(1,3,3)
        img = log_px-torch.min(log_px)
        img/=torch.max(img)
        plt.imshow(img.reshape(len(x1), len(x2)), extent=(-1.2, 2.1, -1.2, 2.1))
        plt.axis((-1.2, 2.1, -1.2, 2.1))
        plt.xticks((-1.2, 2.1))
        plt.yticks((-1.2, 2.1))
        plt.xlabel("$\mathbf{x}_1$", labelpad=-20)
        plt.ylabel("$\mathbf{x}_2$", labelpad=-30)
        plt.title("$p_{\\theta}(\mathbf{x}), step=%d$" % (step+1))

        plt.margins(0, 0)
        if args.out_file is not None:
            plt.savefig("%s-step-%d.png" % (args.out_file,step+1), bbox_inches='tight', pad_inches=0)
        plt.show()


if args.out_file is not None:
    torch.save(flow.state_dict(), ("%s.pt") % (args.out_file))
    from scipy.io import savemat
    savemat(("%s.mat") % (args.out_file), {"his":his})