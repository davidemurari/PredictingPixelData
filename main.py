#Importing necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

from utils import relu_squared
from create_dataset import get_train_test_split
from generate_network import get_network
from training import train as train_network
from generate_plots import generate_gif_predicted, generate_gif_true, generate_gif_error
from generate_plots import generate_error_plots

from get_data import download_data #downloads the missing data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Setting the parameters for the plots
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['font.size']=45
matplotlib.rcParams['font.family']= 'ptm' #'Times New Roman

torch.manual_seed(7)
np.random.seed(7)

#Asking the parameters to the user
check = True
while check:
    pde_name = input("Enter the pde you want (choose among 'linadv', 'heat', 'fisher'):\n")
    if pde_name in ['linadv','heat','fisher']:
        check = False
    else:
        print("Wrong name, please type it again")

#Set to True to use projected Euler, to False to use explicit Euler
conserve_norm = None
if pde_name=='linadv':
    conserve_norm = input("Do you want to conserve the norm while training ('yes', 'no'):\n")
    if conserve_norm=="yes":
        conserve_norm = True
    elif conserve_norm=="no":
        conserve_norm = False
    else:
        print("Not a recognized input. We thus set conserve_norm to False and train with explicit Euler")

#Set to pretrained to use the models trained for the paper
train = input("Do you want to train a new model or use a pre-trained one? ('train','pretrained'):\n")
if train=="train":
    train = True
    timesteps = 6
    while timesteps>5 or timesteps<1:
        timesteps = int(input("Enter how many timesteps you want in the training data (1<=t<=5):\n"))
        if timesteps>5 or timesteps<1:
            print("Wrong range. The data is available only for 1<=timesteps<=5")
elif train=="pretrained":
    train = False
    timesteps = 5

download_data(pde_name)

#Split of the data points into train and test set
trainset, testset = get_train_test_split(pde_name,timesteps=timesteps,device=device)

#Create the model
model = get_network(pde_name,preserve_norm=conserve_norm)
model.to(device);

#Define the training parameters
dim = 100
lr = 1e-2
epochs = 100
batch_size = 32
weight_decay = 0

#Create the dataloaders splitting the dataset into batches
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=30,shuffle=True,num_workers=0) 

if train:
    #Train the model
    train_network(model,lr,weight_decay,epochs,trainloader,timesteps=timesteps)
   
    #Save the trained model
    if pde_name=="linadv":
        if conserve_norm:
            torch.save(model.state_dict(),f"trained_model_{pde_name}_conserved.pt")
        else:
            torch.save(model.state_dict(),f"trained_model_{pde_name}_nonConserved.pt")
    else:
        torch.save(model.state_dict(),f"trained_model_{pde_name}.pt")
else:
    #Load the pre-trained models
    if pde_name=="linadv":
        if conserve_norm:
            model.load_state_dict(torch.load(f"pretrained_models/trained_model_{pde_name}_conserved.pt",map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.load(f"pretrained_models/trained_model_{pde_name}_nonConserved.pt",map_location=torch.device(device)))
    else:
        model.load_state_dict(torch.load(f"pretrained_models/trained_model_{pde_name}.pt",map_location=torch.device(device)))


#Plotting part
showPlots = input("Write yes to plot the errors: ")=="yes"

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
        generate_error_plots(pde_name,model,testloader,conserve_norm)
    else:
        generate_error_plots(pde_name,model,testloader)