import torch
import torch.nn as nn

#Class that defines the relu squared activation function
class relu_squared(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    return torch.relu(x)**2