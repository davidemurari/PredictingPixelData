import os
import warnings

import torch
import torch.nn as nn
import numpy as np
from utils import relu_squared
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self,is_linear=False,zero_sum_first=False,bias=False,pde_name='linadv',kernel_size=5,nlayers=3,dt=0.1,preserve_norm=False,requires_grad_second=True):
        super().__init__()
        
        self.dt = dt
        self.nlayers = nlayers
        
        self.is_linear = is_linear
        self.preserve_norm = preserve_norm
        self.zero_sum = zero_sum_first
        self.pde_name = pde_name
        
        relu2 = relu_squared()
        
        if kernel_size==5:
            pad = 2
        elif kernel_size==7:
            pad = 3
        else:
            pad = 1
        
        self.pad = pad
        self.circular_padding = lambda x : F.pad(x, (self.pad,self.pad,self.pad, self.pad), mode='circular')
        pad_mode = 'circular'
        
        if not pde_name=="linadv":
            self.F = nn.Sequential(
                    nn.Conv2d(1,2,kernel_size=kernel_size,padding=pad,padding_mode=pad_mode,bias=bias),
                    nn.ReLU(),
                    nn.Conv2d(2,1,kernel_size=1,bias=False)
                )
            self.F[-1].weight.data = torch.tensor([1.,-1.]).reshape(*self.F[-1].weight.data.shape)
            self.F[-1].weight.requires_grad = requires_grad_second
            
        else: 
            num_chans = 2
            self.K1 = nn.Conv2d(1,num_chans,kernel_size=kernel_size,padding=pad,padding_mode=pad_mode,bias=bias)
            self.K2 = nn.Conv2d(num_chans,1,kernel_size=1,bias=False)
            self.K2.weight.data = torch.ones_like(self.K2.weight.data)
            self.A = nn.Conv2d(1,1,kernel_size=kernel_size,padding=pad,padding_mode=pad_mode,bias=False)
            
    def energy(self,U):
        return self.K2(torch.relu(self.K1(U))**2).reshape(len(U),-1).sum(dim=1)
    
    def lagrange_multiplier(self,U,energy_0):
        grad = self.gradient(U)
        fact = -1/torch.vmap(torch.trace)(torch.einsum('iokj,iokl->ijl',grad,grad)).unsqueeze(1)
        return fact * (self.energy(U)-energy_0).unsqueeze(1)
    
    def gradient(self,U):
        k1 = torch.relu(self.K1(U))
        lam = self.K2.weight.data*k1
        k1t = F.conv_transpose2d(self.circular_padding(lam),self.K1.weight,padding=4,stride=1)
        return k1t
    
    def vecField(self,U):
        if self.pde_name=="linadv":
            grad = self.gradient(U)
            A = self.A(grad)
            At = F.conv_transpose2d(self.circular_padding(grad),self.A.weight,padding=4,stride=1)
            return A#-At
        else:
            return self.F(U)
    
    def forward(self,U,no_initial=[]):        
        if self.preserve_norm:
            energy_0 = self.energy(U)
        if self.zero_sum:
            self.F[0].weight.data -= torch.mean(self.F[0].weight.data,dim=(2,3)).view(-1,1,1,1)
        
        for _ in range(self.nlayers):
            if self.preserve_norm:
                U = U + self.dt/self.nlayers * self.vecField(U)
                #U = U * (no_initial / torch.linalg.norm(U.reshape(len(U),-1),ord=2,keepdims=True)).view(-1,1,1,1)
                lam = self.lagrange_multiplier(U,energy_0)
                U = U + lam.view(-1,1,1,1) * self.gradient(U)
            else: 
                U = U + self.dt/self.nlayers * self.vecField(U)              
        return U