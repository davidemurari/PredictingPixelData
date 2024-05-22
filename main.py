#Importing necessary packages
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from scripts.generate_plots import *
from scripts.networkArchitecture import network
from scripts.get_data import download_data
from scripts.create_dataset import get_train_test_split
from scripts.training import train_network

if torch.cuda.is_available():
    device_name = 'cuda:0'
elif torch.backends.mps.is_available():
    device_name = 'mps'
else:
    device_name = 'cpu'
device = torch.device(device_name)

torch.manual_seed(7)
np.random.seed(7)

dtype = torch.float32
np_dtype = np.float32

#Asking the parameters to the user
check = True
while check:
    pde_name = input("Enter the pde you want (choose among 'linadv', 'heat', 'fisher'):\n")
    if pde_name in ['linadv','heat','fisher']:
        check = False
    else:
        print("Wrong name, please type it again")

#Set to True to use projected Euler, to False to use explicit Euler
preserve_norm = False
if pde_name=='linadv':
    preserve_norm = input("Type y if you want to conserve the norm while training, any other key otherwise:\n")
    if preserve_norm=="y":
        preserve_norm = True
    else:
        preserve_norm = False

#Create the model
timesteps = 5
nlayers = 3
dim = 100
lr = 5e-3
epochs = 300
batch_size = 32

dim = 100 #Size of the input matrices (100x100)
alpha = .01 #Diffusivity constant
xx = np.linspace(0,1,dim) #Spatial discretization of [0,1]
dx = xx[1]-xx[0] #Spatial step

kernel_size = 5
weight_decay = 0.
is_cyclic = True

is_noise = input("Do you want to consider noise injection? Type y for yes, any other key for no:\n")=="y"
    
download_data(pde_name=pde_name)
    
trainset, testset = get_train_test_split(pde_name,timesteps=timesteps,device=device,dtype=dtype)
dt = 0.1 if pde_name=='linadv' else 0.24 * dx**2 / alpha #Temporal step

config = {
    "dt":dt,
    "dim":dim, 
    "timesteps": timesteps,
    "learning_rate":lr, 
    "preserve_norm":preserve_norm, 
    "epochs":epochs, 
    "batch_size":batch_size, 
    "weight_decay":weight_decay,
    "optimizer":"adam",
    "n_layers":nlayers,
    "kernel_size":kernel_size,
    "is_cyclic_scheduler":is_cyclic,
    "is_added_noise":is_noise
}

print("Current test with : ",pd.DataFrame.from_dict(config,orient='index',columns=["Value"]))

#Create the dataloaders splitting the dataset into batches
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=30,shuffle=True,num_workers=0) 

train = input("Type y to train a new network, any other key to use a pre-trained model:\n")=="y"

model = network(pde_name=pde_name,
                kernel_size=kernel_size,
                nlayers=nlayers,
                dt=dt,
                preserve_norm=preserve_norm,
                dtype=dtype
                )
model.to(device);

noise_tag = "Noise" if is_noise else "NoNoise"
if pde_name=="linadv":
    pde_tag = "Linadv"
elif pde_name=="heat":
    pde_tag = "Heat"
else:
    pde_tag = "Fisher"

if train:
    #Train the model
    loss = train_network(model,lr,epochs,trainloader,timesteps,is_cyclic,is_noise,device)

    #Save the trained model
    if pde_name=="linadv":
        if preserve_norm:
            torch.save(model.state_dict(),f"pretrained_models/{pde_name}Preserve{noise_tag}.pt")
        else:
            torch.save(model.state_dict(),f"pretrained_models/{pde_name}NoPreserve{noise_tag}.pt")
    else:
        torch.save(model.state_dict(),f"pretrained_models/{pde_name}{noise_tag}.pt")
else:
    #Load the pre-trained models
    if pde_name=="linadv":
        if preserve_norm:
            model.load_state_dict(torch.load(f"pretrained_models/{pde_name}Preserve{noise_tag}.pt",map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.load(f"pretrained_models/{pde_name}NoPreserve{noise_tag}.pt",map_location=torch.device(device)))
    else:
        model.load_state_dict(torch.load(f"pretrained_models/{pde_name}{noise_tag}.pt",map_location=torch.device(device)))

#Plotting part
showPlots = input("Write y to plot the errors: ")=="y"

if showPlots:
    
    #Move to evaluation mode and show the results on test points
    model.to('cpu')
    model.eval();

    X,Y = next(iter(testloader))

    X = X[0].to('cpu')
    Y = Y[0].to('cpu')

    timesteps_test = len(Y)

    #Generation of the plots reported also in the paper
    #generate_gif_predicted(pde_name,model,X,timesteps_test)
    #generate_gif_true(pde_name,X,Y,timesteps_test)
    #generate_gif_error(pde_name,model,X,Y,timesteps_test)

    if pde_name=='linadv':
        generate_error_plots(pde_name,model,testloader,preserve_norm,is_noise=is_noise)
    else:
        generate_error_plots(pde_name,model,testloader,is_noise=is_noise)