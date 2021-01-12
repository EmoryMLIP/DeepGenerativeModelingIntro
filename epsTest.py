import torch

def epsTest(X, Y, eps=1e-1):
    """
    Test for equal distributions suggested in Székely, G. J., InterStat, M. R., 2004. (n.d.).
    Testing for equal distributions in high dimension. Personal.Bgsu.Edu.

    :param X: Samples from first distribution
    :param Y: Samples from second distribution
    :param eps: conditioning paramter
    :return:
    """
    nx = X.shape[0]
    ny = Y.shape[0]

    X = X.view(nx, -1)
    Y = Y.view(ny, -1)

    sX = torch.norm(X, dim=1) ** 2;
    sY = torch.norm(Y, dim=1) ** 2;

    CXX = sX.unsqueeze(1) + sX.unsqueeze(0) - 2 * X @ X.t()
    CXX = torch.sqrt(CXX + eps)

    CYY = sY.unsqueeze(1) + sY.unsqueeze(0) - 2 * Y @ Y.t()
    CYY = torch.sqrt(CYY + eps)

    CXY = sX.unsqueeze(1) + sY.unsqueeze(0) - 2 * X @ Y.t()
    CXY = torch.sqrt(CXY + eps)

    D = (nx * ny) / (nx + ny) * (2.0 / (nx * ny) * torch.sum(CXY)
                                 - 1.0 / nx ** 2 * (torch.sum(CXX)) - 1.0 / ny ** 2 * (torch.sum(CYY)));

    return D / (nx + ny)