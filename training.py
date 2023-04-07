import torch
import torch.nn as nn
import numpy as np

import os
    
def train(model,lr,weight_decay,epochs,trainloader,timesteps=5):
    
    device = model.layer1.weight.device
    criterion = nn.MSELoss()
    
    #We define the timesteps to perform while training in a progressive manner
    if timesteps>3:
        listSteps = [timesteps-2,timesteps-1,timesteps]
    else:
        listSteps = [timesteps]
    
    #Training loop that pre-trains the model on simpler problems, where timeMax
    #is smaller and keeps increasing throughout the loop.
    for timeMax in listSteps:

        print(f"Training with {timeMax} timesteps")

        #Definition of optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        lossVal = 1.
        epoch = 1

        #Loop over the epochs for a fixed number of timesteps used as a comparison
        #with the network predictions (i.e. fixed timeMax)
        count = 0
        while epoch < epochs and lossVal>1e-20:
            losses = []
            running_loss = 0
            
            for i, inp in enumerate(trainloader):
                inputs, labels = inp
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()

                loss = 0.
                
                res = inputs
                for tt in range(timeMax):
                    res = model(res)
                    loss += criterion(res,labels[:,tt:tt+1]) / timeMax
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                running_loss += loss.item()
                
                count += len(inputs)
                
                if i%15 == 0 and i > 0:
                    print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 15)
                    running_loss = 0.0
            
            lossVal = loss.item()
            epoch += 1
            #Learning rate scheduler for the traininig loop
            scheduler.step()
        #Adapt the starting learning rate for the next timeMax value
        lr=lr/5
    print('Training Done')