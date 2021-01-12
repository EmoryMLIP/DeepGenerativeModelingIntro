import torch
from torch import nn
import argparse
import numpy as np
from torch import distributions
from sklearn import datasets
import matplotlib.pyplot as plt
import sys,os

ot_flow_dir = '/Users/lruthot/Google Drive/OT-Flow/'
if not os.path.exists(ot_flow_dir):
    raise Exception("Cannot find OT-Flow in %s" %(ot_flow_dir))
sys.path.append(os.path.dirname(ot_flow_dir))

from src.plotter import plot4
from src.OTFlowProblem import *

device = "cpu"

parser = argparse.ArgumentParser('OTFlow')
parser.add_argument("--batch_size"    , type=int, default=1024, help="batch size")
parser.add_argument("--noise"    , type=int, default=0.05, help="noise of moons")
parser.add_argument("--width"    , type=int, default=32, help="width of neural net")
parser.add_argument('--alph'  , type=str, default='1.0,10.0,5.0',help="alph[0]-> weight for transport costs, alph[1] and alph[2]-> HJB penalties")
parser.add_argument("--nTh"    , type=int, default=2, help="number of layers")
parser.add_argument("--nt"    , type=int, default=4, help="number of time steps in training")
parser.add_argument("--nt_val"    , type=int, default=8, help="number of time steps in validation")
parser.add_argument("--num_steps"    , type=int, default=10000, help="number of training steps")
parser.add_argument("--plot_interval"    , type=int, default=500, help="plot solution every so many steps")
parser.add_argument("--out_file", type=str, default=None, help="base filename saving trained model (extension .pt), history (extension .mat), and intermediate plots (extension .png")

args = parser.parse_args()
args.alph = [float(item) for item in args.alph.split(',')]


def compute_loss(net, x, nt):
    Jc , cs = OTFlowProblem(x, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


net = Phi(nTh=args.nTh, m=args.width, d=2, alph=args.alph)
optim = torch.optim.Adam(net.parameters(), lr=1e-2) # lr=0.04 good

his = np.zeros((0,4))


print((3*"--" + "device=%s, nTh=%d, width=%d, batch_size=%d, num_steps=%d" + 3*"--") % (device, args.nTh, args.width, args.batch_size, args.num_steps, ))

out_dir = "results/OTFlow-noise-%1.5f-nTh-%d-width-%d" % (args.noise, args.nTh, args.width)

if args.out_file is not None:
    import os
    out_dir, fname = os.path.split(args.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print((3*"--" + "out_file: %s" + 3*"--") % (args.out_file))

print((5*"%7s    ") % ("step","J", "J_L", "J_ML","J_HJB"))

train_J = 0.0
train_L = 0.0
train_JML = 0.0
num_step = 0
train_HJB = 0.0

for step in range(args.num_steps):

    x = torch.tensor(datasets.make_moons(n_samples=args.batch_size, noise=args.noise)[0], dtype=torch.float32)
    optim.zero_grad()
    loss, costs = compute_loss(net, x, nt=args.nt)
    loss.backward()
    optim.step()

    train_J += loss.item()
    train_L += costs[0].item()
    train_JML += costs[1].item()
    train_HJB += costs[2].item()
    num_step += 1

    if (step + 1) % args.plot_interval == 0:
        train_J /= num_step
        train_JML /= num_step
        train_L /= num_step
        train_HJB /= num_step


        print(("%06d   " + 4*"%1.4e  ") %
              (step + 1, train_J, train_L, train_JML, train_HJB))
        his = np.vstack([his, [train_J, train_L, train_JML, train_HJB]])
        train_J = 0.0
        train_L = 0.0
        train_JML = 0.0
        num_step = 0
        train_HJB = 0.0

        with torch.no_grad():
            nSamples = 10000
            xs = torch.tensor(datasets.make_moons(n_samples=nSamples, noise=args.noise)[0], dtype=torch.float32)
            zs = torch.randn(nSamples, 2)  # sampling from the standard normal (rho_1)
            if args.out_file is not None:
                plot4(net, xs, zs, args.nt_val, "%s-step-%d.png" % (args.out_file,step+1), doPaths=True)
                plt.show()

if args.out_file is not None:
    torch.save(net.state_dict(), ("%s.pt") % (args.out_file))
    from scipy.io import savemat
    savemat(("%s.mat") % (args.out_file), {"his":his})