import torch
import torch.nn as nn
import numpy as np

import os
    
def train_network(model,lr,epochs,trainloader,timesteps=3,is_cyclic=True,is_noise=True,device='cpu'):
    
    criterion = nn.MSELoss()
    
    best_loss = 100.
    
    if timesteps>3:
        listSteps = [timesteps-2,timesteps-1,timesteps]
    else:
        listSteps = [timesteps]
    
    #Training loop that pre-trains the model on simpler problems, where timeMax
    #is smaller and keeps increasing throughout the loop.
    for max_t in listSteps:
        
        print(f"Training with {max_t} timesteps")

        #Definition of optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        if is_cyclic:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=10000, mode='exp_range',cycle_momentum=False)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=int(0.45*epochs))
        epoch = 1
    
        #Loop over the epochs for a fixed number of timesteps used as a comparison
        #with the network predictions (i.e. fixed max_t)
        while epoch < epochs:
            
            for i, inp in enumerate(trainloader):
                inputs, labels = inp
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                loss = 0.
                
                epsilon = 1e-2
                
                res = inputs.clone()
                
                no_initial = torch.linalg.norm(res.reshape(len(res),-1),dim=1,ord=2,keepdims=True)
                
                for tt in range(max_t):
                    
                    #Compute the noise to add to the initial condition
                    if is_noise:
                        noise = (torch.rand_like(inputs)*2*epsilon-epsilon)
                    else:
                        noise = 0.
                    #Add the noise to the input
                    if tt==0:
                        res = model(res + noise,no_initial)
                    else:
                        res = model(res)
                    #Increment the loss term with the current contribution
                    loss += criterion(res,labels[:,tt:tt+1]) / max_t
                
                loss.backward()
                
                optimizer.step()
                if is_cyclic:
                    scheduler.step()
            
            if is_cyclic==False:
                scheduler.step()
            
            epoch += 1
            
            if epoch%5==0:
                print(f'Loss [{epoch}](epoch): ', loss.item())
            
            if epoch>int(0.85*epochs) and max_t==timesteps-1:
                if loss.item()<best_loss:
                    torch.save(model.state_dict(), "pretrained_models/best_model.pt")
                    best_loss = loss.item()
        lr = lr/2
        
    print('Training Done')