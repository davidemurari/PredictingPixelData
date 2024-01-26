from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torch
import os

#Class generating the dataset given the data points
class dataset(Dataset):
  def __init__(self,x,y,device,np_dtype):
    self.x = torch.from_numpy(x.astype(np_dtype)).to(device)
    self.y = torch.from_numpy(y.astype(np_dtype)).to(device)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length

#Method splitting the dataset into train and test sets
def get_train_test_split(pde_name,device,timesteps=5,dtype=torch.float32):
    
    np_dtype = np.float32 if dtype==torch.float32 else np.float64
    
    #Load the data points
    with open(f'data/data_{pde_name}.pickle','rb') as file:
      data_train = pickle.load(file)
    with open(f'data/data_{pde_name}_verification.pickle','rb') as file:
      data_test = pickle.load(file)
    
    #Select the initial conditions with norm larger than 10
    #that are then used to train the network for the Fisher equation
    if pde_name=="fisher":
      new_data_train = []
      new_data_test = []
      for i in range(len(data_train)):
        if np.linalg.norm(data_train[i][0].reshape(-1),ord=2).item()>10:
          new_data_train.append(data_train[i])
      
      for i in range(len(data_test)):
        if np.linalg.norm(data_test[i][0].reshape(-1),ord=2).item()>10:
          new_data_test.append(data_test[i])
      
      data_train = new_data_train
      data_test = new_data_test
    
    
    #Split the loaded data into training and testing sets
    dim1,dim2 = data_train[1][0].shape
    timesteps_test = len(data_test[1])-1 #we subtract one because we also have the IC
    
    n_train = len(data_train)
    n_test = len(data_test)
    input_train = np.zeros((n_train,1,dim1,dim2))
    label_train = np.zeros((n_train,timesteps,dim1,dim2))
    input_test = np.zeros((n_test,1,dim1,dim2))
    label_test = np.zeros((n_test,timesteps_test,dim1,dim2))
    
    #Store the initial condition into the input variables
    #and the remaining updates into the label variables.
    for i in range(n_train):
        input_train[i,0] = data_train[i][0]
        for j in range(timesteps):
            label_train[i,j] = data_train[i][j+1]
    for i in range(n_test):
        input_test[i,0] = data_test[i][0]
        for j in range(timesteps_test):
            label_test[i,j] = data_test[i][j+1]
    
    #Create the dataloaders given the obtained splitting
    trainset = dataset(input_train,label_train,device,np_dtype=np_dtype)
    testset = dataset(input_test,label_test,device,np_dtype=np_dtype)
    
    return trainset, testset