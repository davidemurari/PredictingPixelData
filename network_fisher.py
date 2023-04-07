import torch
import torch.nn as nn
import numpy as np
from utils import relu_squared

class network(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 100 #Size of the input matrices (100x100)
        alpha = .01 #Diffusivity constant
        xx = np.linspace(0,1,dim) #Spatial discretization of [0,1]
        dx = xx[1]-xx[0] #Spatial step
        self.dt = 0.24 * dx**2 / alpha #Temporal step

        #Activation functions and number of layers 
        self.nl1 = nn.ReLU()
        self.nl2 = relu_squared()
        self.nlayers = 3
        
        #Layers for the linear and non-linear parts
        self.layer1 = nn.Conv2d(1,8,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='zeros')
        self.layer2 = nn.Conv2d(8,16,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='zeros')
        self.layer3 = nn.Conv2d(16,16,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='zeros')
        self.layer4 = nn.Conv2d(16,8,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='zeros')
        self.layer5 = nn.Conv2d(8,1,kernel_size=3,padding=1,stride=1,bias=False,padding_mode='zeros')

        self.quadratic_int1 = nn.Conv2d(1,16,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='zeros')
        self.quadratic_int2 = nn.Conv2d(16,1,kernel_size=3,padding=1,stride=1,bias=False,padding_mode='zeros')

        #Weight initialization
        nn.init.orthogonal_(self.layer1.weight)
        nn.init.orthogonal_(self.layer2.weight)
        nn.init.orthogonal_(self.layer3.weight)
        nn.init.orthogonal_(self.layer4.weight)
        nn.init.orthogonal_(self.layer5.weight)
        nn.init.orthogonal_(self.quadratic_int1.weight)
        nn.init.orthogonal_(self.quadratic_int2.weight)

        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
        nn.init.zeros_(self.layer4.bias)
        nn.init.zeros_(self.quadratic_int1.bias)

    #Evaluation of the approximate spatially discretized PDE
    def F(self,U):
        lin = self.nl1(self.layer1(U))
        lin = self.nl1(self.layer2(lin))
        lin = self.nl1(self.layer3(lin))
        lin = self.nl1(self.layer4(lin))
        lin = self.layer5(lin)

        return lin + self.quadratic_int2(self.nl2(self.quadratic_int1(U)))

    #Forward stepping in time
    def forward(self,U):
        for i in range(self.nlayers):
          U = U + self.dt / self.nlayers * self.F(U)
        return U