import torch
import numpy as np
import torch.nn as nn


def ToLabel(E):
    fgs = np.argmax(E, axis=1).astype(np.float32)
    return fgs.astype(np.uint8)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def SaliencyStructureConsistency(x, y, alpha):
    ssim = torch.mean(SSIM(x,y))
    l1_loss = torch.mean(torch.abs(x-y))
    loss_ssc = alpha*ssim + (1-alpha)*l1_loss
    return loss_ssc


def SaliencyStructureConsistencynossim(x, y):
    l1_loss = torch.mean(torch.abs(x-y))
    return l1_loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




