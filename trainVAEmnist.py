from vae import *
from torch import distributions
import argparse

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## load MNIST
parser = argparse.ArgumentParser('VAE')
parser.add_argument("--batch_size"    , type=int, default=128, help="batch size")
parser.add_argument("--q"    , type=int, default=2, help="latent space dimension")
parser.add_argument("--width_enc"    , type=int, default=4, help="width of encoder")
parser.add_argument("--width_dec"    , type=int, default=4, help="width of decoder")
parser.add_argument("--num_epochs"    , type=int, default=2, help="number of epochs")
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

test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


from modelMNIST import  Encoder, Generator
g = Generator(args.width_dec,args.q)
e = Encoder(args.width_enc,args.q)

vae = VAE(e,g).to(device)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=1e-3, weight_decay=1e-5)

his = np.zeros((args.num_epochs,6))

print((3*"--" + "device=%s, q=%d, batch_size=%d, num_epochs=%d, w_enc=%d, w_dec=%d" + 3*"--") % (device, args.q, args.batch_size, args.num_epochs, args.width_enc, args.width_dec))

if args.out_file is not None:
    import os
    out_dir, fname = os.path.split(args.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print((3*"--" + "out_file: %s" + 3*"--") % (args.out_file))

print((7*"%7s    ") % ("epoch","Jtrain","pzxtrain","ezxtrain","Jval","pzxval","ezxval"))


for epoch in range(args.num_epochs):
    vae.train()

    train_loss = 0.0
    train_pzx = 0.0
    train_ezx = 0.0
    num_ex = 0
    for image_batch, _ in train_dataloader:
        image_batch = image_batch.to(device)

        # take a step
        loss, pzx, ezx,gz,mu = vae.ELBO(image_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update history
        train_loss += loss.item()*image_batch.shape[0]
        train_pzx += pzx*image_batch.shape[0]
        train_ezx += ezx*image_batch.shape[0]
        num_ex += image_batch.shape[0]

    train_loss /= num_ex
    train_pzx /= num_ex
    train_ezx /= num_ex

    # evaluate validation points
    vae.eval()
    val_loss = 0.0
    val_pzx = 0.0
    val_ezx = 0.0
    num_ex = 0
    for image_batch, label_batch in test_dataloader:
        with torch.no_grad():
            image_batch = image_batch.to(device)
            # vae reconstruction
            loss, pzx, ezx, gz, mu = vae.ELBO(image_batch)
            val_loss += loss.item() * image_batch.shape[0]
            val_pzx += pzx * image_batch.shape[0]
            val_ezx += ezx * image_batch.shape[0]
            num_ex += image_batch.shape[0]

    val_loss /= num_ex
    val_pzx/= num_ex
    val_ezx/= num_ex

    print(("%06d   " + 6*"%1.4e  ") %
          (epoch + 1, train_loss, train_pzx, train_ezx, val_loss, val_pzx, val_ezx))

    his[epoch,:] = [train_loss, train_pzx, train_ezx, val_loss, val_pzx, val_ezx]

if args.out_file is not None:
    torch.save(vae.g.state_dict(), ("%s-g.pt") % (args.out_file))
    torch.save(vae.state_dict(), ("%s.pt") % (args.out_file))
    from scipy.io import savemat
    savemat(("%s.mat") % (args.out_file), {"his":his})