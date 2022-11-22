import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CosSimilarity(nn.Module):
    def __init__(self, use_scale=True):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.use_scale = use_scale

    def forward(self, x, y):
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - self.cos(x, y)