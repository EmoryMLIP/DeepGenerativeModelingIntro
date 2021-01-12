import torch
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser('DCGAN')
parser.add_argument("--batch_size"    , type=int, default=64, help="batch size")
parser.add_argument("--q"    , type=int, default=2, help="latent space dimension")
parser.add_argument("--width_disc"    , type=int, default=32, help="width of discriminator")
parser.add_argument("--width_dec"    , type=int, default=32, help="width of decoder")
parser.add_argument("--num_steps"    , type=int, default=50, help="number of training steps")
parser.add_argument("--plot_interval"    , type=int, default=5, help="plot solution every so many steps")
parser.add_argument("--init_g", type=str, default=None, help="path to .pt file that contains weights of a trained generator")
parser.add_argument("--out_file", type=str, default=None, help="base filename saving trained model (extension .pt), history (extension .mat), and intermediate plots (extension .png")

args = parser.parse_args()

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

img_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

from modelMNIST import Generator, Discriminator
g = Generator(args.width_dec,args.q).to(device)
d = Discriminator(args.width_disc,useSigmoid=True).to(device)

optimizer_g = torch.optim.Adam(params=g.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(params=d.parameters(), lr=0.0002, betas=(0.5, 0.999))

his = np.zeros((0,3))

if args.init_g is not None:
    print("initialize g with weights in %s" % args.init_g)
    g.load_state_dict(torch.load(args.init_g))

print((3*"--" + "device=%s, q=%d, batch_size=%d, num_steps=%d, w_disc=%d, w_dec=%d," + 3*"--") % (device, args.q, args.batch_size, args.num_steps, args.width_disc, args.width_dec))
if args.out_file is not None:
    import os
    out_dir, fname = os.path.split(args.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print((3*"--" + "out_file: %s" + 3*"--") % (args.out_file))
print((4*"%7s    ") % ("step","J_GAN","J_Gen","ProbDist"))

from epsTest import epsTest

train_JGAN = 0.0
train_JGen = 0.0
train_epsTest = 0.0
num_ex = 0

def inf_train_gen():
    while True:
        for images, targets in enumerate(train_dataloader):
            yield images,targets

get_true_images = inf_train_gen()

for step in range(args.num_steps):
    g.train()
    d.train()
    # update discriminator using - J_GAN = - E_x [log(d(x)] - E_z[1-log(d(g(z))]
    x = get_true_images.__next__()[1][0]
    x = x.to(device)
    dx = d(x)
    z = torch.randn((x.shape[0],args.q),device=device)
    gz = g(z)
    dgz = d(gz)
    J_GAN = - torch.mean(torch.log(dx)) - torch.mean(torch.log(1-dgz))
    optimizer_d.zero_grad()
    J_GAN.backward()
    optimizer_d.step()

    # update the generator using J_Gen = - E_z[log(d(g(z))]
    optimizer_g.zero_grad()
    z = torch.randn((x.shape[0], args.q), device=device)
    gz = g(z)
    dgz = d(gz)
    # J_Gen = -torch.mean(torch.log(dgz))
    J_Gen = torch.mean(torch.log(1-dgz))
    J_Gen.backward()
    optimizer_g.step()

    # update history
    train_JGAN -= J_GAN.item()*x.shape[0]
    train_JGen += J_Gen.item()*x.shape[0]
    train_epsTest += epsTest(gz.detach(), x)
    num_ex += x.shape[0]

    if (step + 1) % args.plot_interval == 0:
        train_JGAN /= num_ex
        train_JGen /= num_ex


        print(("%06d   " + 3*"%1.4e  ") %
              (step + 1, train_JGAN, train_JGen, train_epsTest))
        his = np.vstack([his, [train_JGAN, train_JGen, train_epsTest]])

        plt.Figure()
        img = gz.detach().cpu()
        img -= torch.min(img)
        img /=torch.max(img)
        plt.imshow(torchvision.utils.make_grid(img, 16, 5,pad_value=1.0).permute((1, 2, 0)))
        plt.title("trainDCGANmnist: step=%d" % (step + 1))
        if args.out_file is not None:
            plt.savefig(("%s-step-%d.png") % (args.out_file, step + 1))
        plt.show()

        train_JGAN = 0.0
        train_JGen = 0.0
        train_epsTest = 0.0
        num_ex = 0

if args.out_file is not None:
    torch.save(g.state_dict(), ("%s-g.pt") % (args.out_file))
    torch.save(d.state_dict(), ("%s-d.pt") % (args.out_file))

    from scipy.io import savemat
    savemat(("%s.mat") % (args.out_file), {"his":his})

plt.Figure()
plt.subplot(1,2,1)
plt.plot(his[:,0:2])
plt.legend(("JGAN","JGen"))
plt.title("GAN Objectives")
plt.subplot(1,2,2)
plt.plot(his[:,2])
plt.title("epsTest")
if args.out_file is not None:
    plt.savefig(("%s-his.png") % (args.out_file))
plt.show()