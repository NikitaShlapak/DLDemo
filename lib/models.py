import torch
torch.random.manual_seed(42)
import random
random.seed(42)
from torch import nn


class SimpleModel(nn.Module):

    def __init__(self, n_layers:int, input_num:int, shapes:list[int]=None):
        assert(n_layers >= 1)
        assert(input_num >= 1)
        assert len(shapes) == n_layers or shapes is None
        super().__init__()


        if shapes is None:
            shapes = [input_num*2 for _ in range(n_layers)]
        shapes = [input_num] + shapes

        layers = []
        for i in range(len(shapes)-1):
            layers.append(nn.Linear(shapes[i], shapes[i+1]))
            layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

