import matplotlib.pyplot as plt
import torch


def plot_x(xs,domain=(0,1,0,1)):
    plt.plot(xs[:, 0], xs[:, 1], "bs")
    plt.axis("square")
    plt.axis(domain)
    plt.xticks(domain[0:2])
    plt.yticks(domain[2:])
    plt.xlabel("$\mathbf{x}_1$", labelpad=-20)
    plt.ylabel("$\mathbf{x}_2$", labelpad=-30)


def plot_z(zs):
    plt.plot(zs[:, 0], zs[:, 1], "or")
    plt.axis("square")
    plt.axis((-3.5, 3.5, -3.5, 3.5))
    plt.xticks((-3.5, 3.5))
    plt.yticks((-3.5, 3.5))
    plt.xlabel("$\mathbf{z}_1$", labelpad=-20)
    plt.ylabel("$\mathbf{z}_2$", labelpad=-30)


def plot_px(log_px,domain=(0,1,0,1)):
    px = torch.exp(log_px)
    img = px
    plt.imshow(img.t(), extent=domain,origin='lower')
    plt.axis("square")
    plt.axis(domain)
    plt.xticks(domain[0:2])
    plt.yticks(domain[2:])
    plt.xlabel("$\mathbf{x}_1$", labelpad=-20)
    plt.ylabel("$\mathbf{x}_2$", labelpad=-30)

def plot_pz(zz,domain=(-3.5, 3.5, -3.5, 3.5)):
    plt.hist2d(zz[:,0], zz[:,1],bins=256,range=[[-domain[0], domain[1]], [domain[2], domain[3]]])
    plt.axis("square")
    plt.axis(domain)
    plt.xticks(domain[0:2])
    plt.yticks(domain[2:])
    plt.xlabel("$\mathbf{z}_1$", labelpad=-20)
    plt.ylabel("$\mathbf{z}_2$", labelpad=-30)