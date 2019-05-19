# coding=utf-8
# from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F


# Spatial transformer network forward function
def transformer(x):
    # Spatial transformer localization-network
    localization = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=7),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True),
        nn.Conv2d(8, 10, kernel_size=5),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True)
    )

    # Regressor for the 3 * 2 affine matrix
    fc_loc = nn.Sequential(
        nn.Linear(10 * 3 * 3, 32),
        nn.ReLU(True),
        nn.Linear(32, 3 * 2)
    )

    # Initialize the weights/bias with identity transformation
    fc_loc[2].weight.data.zero_()
    fc_loc[2].bias.data.copy_(
        torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    xs = localization(x)
    xs = xs.view(-1, 10 * 3 * 3)
    theta = fc_loc(xs)
    theta = theta.view(-1, 2, 3)

    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)

    return x