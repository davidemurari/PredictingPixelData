import torch
import torch.nn as nn
from scripts.utils import relu_squared
import torch.nn.functional as F
import warnings

#Definition of the neural network architecture
class network(nn.Module):
    def __init__(self,
                 pde_name='linadv',
                 kernel_size=5,
                 nlayers=3,
                 dt=0.1,
                 preserve_norm=False,
                 dtype=torch.float32):
        super().__init__()
        
        self.dt = dt
        self.nlayers = nlayers
        
        self.preserve_norm = preserve_norm
        self.pde_name = pde_name
        self.correct_vec_field = preserve_norm #we do so only if we want to preserve the norm
                
        if kernel_size==5:
            self.pad = 2
        elif kernel_size==7:
            self.pad = 3
        elif kernel_size==3:
            self.pad = 1
        else:
            warnings.warn("Kernel size not implemented. Set to 5")
            kernel_size = 5
            self.pad = 2

        pad_mode = 'circular'
        
        if pde_name=="heat":
            self.F = nn.Sequential(
                    nn.Conv2d(1,2,kernel_size=kernel_size,padding=self.pad,padding_mode=pad_mode,bias=True,dtype=dtype),
                    nn.ReLU(),
                    nn.Conv2d(2,1,kernel_size=kernel_size,padding=self.pad,padding_mode=pad_mode,bias=False,dtype=dtype)
                )
        elif pde_name=="fisher":
            relu2 = relu_squared()
            self.F = nn.Sequential(
                    nn.Conv2d(1,10,kernel_size=kernel_size,padding=self.pad,padding_mode=pad_mode,bias=True,dtype=dtype),
                    relu2,
                    nn.Conv2d(10,1,kernel_size=kernel_size,padding=self.pad,padding_mode=pad_mode,bias=True,dtype=dtype)
                )
        else: 
            self.F = nn.Sequential(
                nn.Conv2d(1,2,kernel_size=kernel_size,padding=self.pad,padding_mode=pad_mode,bias=True,dtype=dtype),
                nn.ReLU(),
                nn.Conv2d(2,1,kernel_size=kernel_size,padding=self.pad,padding_mode=pad_mode,bias=False,dtype=dtype)
            )
    
    #Evaluation of the approximate spatially discretized PDE
    def vecField(self,U):
        #Unconstrained expression of the vector field
        FF = self.F(U) 
        #Orthogonal projection of the vector field
        if self.correct_vec_field:
            dot_x_f = (U*FF).reshape(len(U),-1).sum(dim=1).reshape(-1,1,1,1)
            dot_x_x = (U*U).reshape(len(U),-1).sum(dim=1).reshape(-1,1,1,1)
            return FF - U * dot_x_f / dot_x_x
        else:
            return FF
    
    def forward(self,U,no_initial=[]):   
        #Compute the initial norm in case it is not provided     
        if self.preserve_norm:
            if len(no_initial)==0:
                no_initial = torch.linalg.norm(U.reshape(len(U),-1),ord=2,dim=1)
        for _ in range(self.nlayers):
            #Unconstrained Euler step
            U = U + self.dt/self.nlayers * self.vecField(U)
            #Projection step to preserve the norm
            if self.preserve_norm:
                U = U * no_initial.view(-1,1,1,1) / torch.linalg.norm(U.reshape(len(U),-1),ord=2,dim=1).view(-1,1,1,1)
        return U