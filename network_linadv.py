import torch
import torch.nn as nn
from utils import relu_squared

class network(nn.Module):
    def __init__(self,preserve_norm=True):
        super().__init__()

        self.dt = 0.1/5 #Temporal step

        #Activation functions and number of layers 
        self.nl1 = nn.ReLU()
        self.nl2 = relu_squared()
        self.nlayers = 2
        
        #Set to true to use projected Euler 
        self.preserve_norm = preserve_norm
        
        #Layers for the linear and non-linear parts
        self.layer1 = nn.Conv2d(1,8,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='circular')
        self.layer2 = nn.Conv2d(8,16,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='circular')
        self.layer3 = nn.Conv2d(16,16,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='circular')
        self.layer4 = nn.Conv2d(16,8,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='circular')
        self.layer5 = nn.Conv2d(8,1,kernel_size=3,padding=1,stride=1,bias=False,padding_mode='circular')

        self.quadratic_int1 = nn.Conv2d(1,8,kernel_size=3,padding=1,stride=1,bias=True,padding_mode='circular')
        self.quadratic_int2 = nn.Conv2d(8,1,kernel_size=3,padding=1,stride=1,bias=False,padding_mode='circular')


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


        self.layer1.weight.data = self.layer1.weight.data - torch.mean(self.layer1.weight.data.reshape(8,1,-1),dim=2).reshape(8,1,1,1)
        self.quadratic_int1.weight.data = self.quadratic_int1.weight.data - torch.mean(self.quadratic_int1.weight.data.reshape(8,1,-1),dim=2).reshape(8,1,1,1)


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

        #Steps of projected Euler
        if self.preserve_norm:
            initNo = torch.linalg.norm(U[:,-1].view(len(U),-1),dim=1,ord=2)
        
        #Steps of (Projected) Euler
        for i in range(self.nlayers):
          U = U + self.dt / self.nlayers * self.F(U)
          if self.preserve_norm:
            finNo = torch.linalg.norm(U.view(len(U),-1),dim=1,ord=2)
            normRatio = initNo / finNo             
            U = torch.einsum('iRjk,i->iRjk',U,normRatio)
        
        return U