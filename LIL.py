import torch
import numpy as np


def L_data(x, y, alpha=1.0, beta=1.0):

    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1  # cross-entropy

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index]  # , :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def L_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




