from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torch
import os

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
def get_train_test_split(pde_name='linadv',timesteps=5,device='cpu',np_dtype=np.float32):
    
    #Load the data points
    if pde_name=="heat":

        try:
          with open(f'data/data_{pde_name}_periodic.pickle','rb') as file:
            data_train = pickle.load(file)
          with open(f'data/data_{pde_name}_periodic_verification.pickle','rb') as file:
            data_test = pickle.load(file)
        
        except:
          try:
            import wget
          except:
            input('To download the data the dataset for heat equation \'wget\' needs to be imported.\n Press enter to agree on dowloading it.') #if you press enter you go on
            os.system('pip install wget')
            import wget

          url = 'https://folk.ntnu.no/jamesij/PredictingPixelData/data_heat_periodic.pickle'
          wget.download(url, 'data/data_heat_periodic.pickle')
          url = 'https://folk.ntnu.no/jamesij/PredictingPixelData/data_heat_periodic_verification.pickle'
          wget.download(url, 'data/data_heat_periodic_verification.pickle')
          
          with open(f'data/data_{pde_name}_periodic.pickle','rb') as file:
            data_train = pickle.load(file)
          with open(f'data/data_{pde_name}_periodic_verification.pickle','rb') as file:
            data_test = pickle.load(file)
    else:
        with open(f'data/data_{pde_name}.pickle','rb') as file:
              data_train = pickle.load(file)
        with open(f'data/data_{pde_name}_verification.pickle','rb') as file:
              data_test = pickle.load(file)
    
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
    #input_train = np.zeros((n_train*2,1,dim1,dim2)) #for the more interesting dataset linadv
    #label_train = np.zeros((n_train*2,timesteps,dim1,dim2)) #for the more interesting dataset linadv
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
   #This is for the 'more interesting' dataset in linear advection 
    '''for i in np.arange(n_train,2*n_train):
        input_train[i,0] = data_train[i-n_train][timesteps]
        for j in range(timesteps):
            label_train[i,j] = data_train[i-n_train][timesteps+j+1]'''
    for i in range(n_test):
        input_test[i,0] = data_test[i][0]
        for j in range(timesteps_test):
            label_test[i,j] = data_test[i][j+1]
    
    #Create the dataloaders given the obtained splitting
    trainset = dataset(input_train,label_train,device,np_dtype)
    testset = dataset(input_test,label_test,device,np_dtype)
    
    return trainset, testset