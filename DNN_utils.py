import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



