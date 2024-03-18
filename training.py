import torch
import torch.nn as nn
import numpy as np

import os
    
def train_network(model,lr,epochs,trainloader,timesteps=3,gamma=1e-4,is_cyclic=True,is_noise=True,device='cpu'):
    
    criterion = nn.MSELoss()
    
    
    for max_t in np.arange(2,timesteps):
        
        print(f"Training with {max_t} timesteps")

        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        if is_cyclic:
            #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=2000, mode='exp_range',cycle_momentum=False)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=10000, mode='exp_range',cycle_momentum=False)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=int(0.45*epochs))
        epoch = 1
    
        while epoch < epochs:
            
            for i, inp in enumerate(trainloader):
                inputs, labels = inp
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                loss = 0.
                
                epsilon = 0.01
                
                res = inputs.clone()
                
                no_initial = torch.linalg.norm(res.reshape(len(res),-1),dim=1,ord=2,keepdims=True)
                sum_initial = torch.sum(res.reshape(len(res),-1),dim=1)
                
                is_lagrange=True
                if epoch<200 and max_t==2:
                    is_lagrange=False
                
                for tt in range(max_t):
                    if is_noise:
                        noise = (torch.rand_like(inputs)*2*epsilon-epsilon)
                    else:
                        noise = 0.
                    if tt==0:
                        res = model(res + noise,no_initial)#,sum_initial)
                    else:
                        res = model(res)
                    loss += criterion(res,labels[:,tt:tt+1]) / max_t
                    
                    
                    if gamma>0:
                        no_current = torch.linalg.norm(res.reshape(len(res),-1),dim=1,ord=2)
                        loss += gamma * criterion(no_current,no_initial) / max_t
                
                def inner(a,b):
                    return (torch.transpose(a,2,3)*b).reshape(len(a),-1).sum(dim=1)
                #loss += 1e-4*torch.mean(inner(inputs,model.A(inputs))**2)
                
                loss.backward()
                
                optimizer.step()
                if is_cyclic:
                    scheduler.step()
            
            if is_cyclic==False:
                scheduler.step()
            
            norm = lambda A : torch.linalg.norm(A.reshape(len(A),-1),dim=1,ord=2)
            epoch += 1
            
            if epoch%5==0:
                print(f'Loss [{epoch}](epoch): ', loss.item())
        
        lr = lr/2
        
    print('Training Done')