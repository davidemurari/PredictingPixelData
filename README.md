# Predictions based on pixel data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11619634.svg)](https://doi.org/10.5281/zenodo.11619634)


This repository is a suppliment to the manuscript **Predictions based on pixel data: Insights from PDEs and finite differences** by _Elena Celledoni, James Jackaman, Davide Murari and Brynjulf Owren_. 

Necessary installed libraries can be installed with the line

> pip install -r requirements.txt

The implementation includes methods that allow to work with the three following PDEs:

- Linear advection : $\partial_{t} u = b\cdot \nabla u = \partial_{x} u+\partial_{y}u$.

- Heat equation : $\partial_{t} u = \alpha \Delta u=\alpha(\partial_{xx}u+\partial_{yy}u)$.

- Fisher equation : $\partial_{t} u = \alpha \Delta u + u(1-u)$.

The main file *main.py* allows to interactively choose which of the three is of interest and the specifics of the simulation.

> **Quickstart:** To run the code as set, run the command
> 
>     python main.py
> 
> and then the user is guided through the choices. An example run is as follows:
> 
> - *Enter the pde you want (choose among 'linadv', 'heat', 'fisher'):* heat
> 
> - *Do you want to train a new model or use a pre-trained one? ('train','pretrained'):* pretrained
> 
> - *Write yes to plot the errors:* yes
> 
> This allows to generate the error plots included in the paper for the heat equation.

##### main.py

After importing the necessary Python libraries, the main script asks the user to set the following parameters:

- *pde_name* : name of the PDE to work with among 'linadv', 'heat' and 'fisher'.

- *conserve_norm* : this is set to None for heat and Fisher equations, while it can be set to True/False for the linear advection one. If True, projection Euler method is used in the training and the network will be norm-preserving. If False, explicit Euler is the method of choice.

- *train* : this is a Boolean variable which if set to True will call the training procedure for the network, otherwise a pre-trained model will be loaded in order to show the results.

Then the training and test sets are generated and the neural network with the chosen parameters is instantiated.

The training parameters one can choose are the learning rate, the training epochs, the number of timesteps, the batch size and the weight decay parameter. They can be set manually from this script. The optimizer and the learning rate scheduling can be replaced in the *train* method contained in the script *training.py*.

If 'cuda' device is available, all the calculations and the storage rely on the gpu. If 'mps' device is available, this is the used device. Otherwise the cpu is used.

After either training or loading the pre-trained networks, it is possible to display the results with the implemented plotting tools. See the script *generate_plots.py* for more details on them.

---

We now describe the scripts included in the repository that are called by the *main.py* script.

##### create_dataset.py

- Class : dataset
  
  - Parameters :
    
    - *x,y* : initial condition of the PDE, time updates.
    - *device* : determines on which device will the object of this class live, 'cpu', 'cuda', or 'mps'.
    - *np_dtype* : determines if the data type to use is float32 or float64.
  - Purpose :
    
    - This class defines the train and test dataloaders.

- Method : get_train_test_split
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied.
    - *timesteps* : decides how many discrete steps are included in the training and test sets.
    - *device* : determines the device on which the train and test loader live.
    - *dtype* : determines the data type to use, which can be *torch.float32* or *torch.float64*.
  
  - Purpose : This method returns the train and test loader obtained by loading the data points from zenodo (see script *get_data.py*) and splitting them into training and test sets. These datasets are then returned as dataloaders thanks to the class *dataset*.

##### networkArchitecture.py

- Class : network

  - Parameters :
    
    - *pde_name* : String variable representing the PDE of interest.
    - *kernel_size* : Integer in {3,5,7} representing the size of the convolutional filter .
    -  *nlayers* : Integer representing how many time substeps to do in the network.
    -  *dt* : Float representing the time step.
    - *preserve_norm* : Boolean variable which if true makes the forward pass be based on projected Euler method. If False the explicit Euler method is used instead.
    - *dtype* : Either *torch.float32* or *torch.float64*, depending on the data type of the inputs.
  
  - Purpose : This method aims to define the neural network architecture.

##### generate_plots.py

- Method : generate_gif_predicted
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied.
    
    - *model* : trained neural network to adopt in order to make predictions.
    
    - *X* : test initial condition for the plot.
    
    - *timesteps_test* : number of performed steps to generate the output .gif file.
  
  - Purpose : This method generates a .gif file showing the time evolution of the initial condition *X* updated by *model* for *timesteps_test* iterations. The result is saved in the folder *savedPlots* as *predicted_{pde_name}.gif*.

- Method : generate_gif_true
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied.
    
    - *X* : test initial condition for the plot.
    
    - *Y* : true time evolution of the initial condition *X* given by the PDE discrete flow map.
    
    - *timesteps_test* : number of performed steps to generate the output .gif file.
  
  - Purpose : This method generates a .gif file showing the time evolution of the initial condition *X* updated by a numerical method solving the PDE with initial condition *X* and for *timesteps_test* steps. The result is saved in the folder *savedPlots* as *true_{pde_name}.gif*.

- Method : generate_gif_error
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied.
    
    - *X* : test initial condition for the plot.
    
    - *model* : trained neural network to adopt in order to make predictions.
    
    - *Y* : true time evolution of the initial condition *X* given by the PDE discrete flow map.
    
    - *timesteps_test* : number of performed steps to generate the output .gif file.
  
  - Purpose : This method generates a .gif file showing the time evolution of the difference between the predictions of *model* and the exact update stored in *Y*. The result is saved in the folder *savedPlots* as *error_{pde_name}.gif*.

- Method : save_test_results
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied.
    
    - *model* : trained neural network to adopt in order to make predictions.
    
    - *testloader* : dataloader for the test set used to compute the error metrics.
    
    - *preserve_norm* : Boolean variable which is by default set to None for heat and Fisher equation, and can be True/False for the advection equation.

    - *is_noise* : Boolean variable determining if the experiment is done with or without noise.
  
  - Purpose : Generate .csv files where the three test metrics are stored. We measure the mean squared error, the relative error and the maximum absolute error for 30 test initial conditions and store in these files the mean of such values. The files are saved in the folder *saved_test_results*.

- Method : generate_error_plots
  
  - Same inputs as *save_test_results*.
  
  - Purpose : After generating the .csv files with the method *save_test_results*, this method displays and saves the correspondent plots. These plots are saved as .pdf files in the folder *saved_plots*. The method plots data available in the folder *saved_test_results*. Thus, to display the comparison between projected Euler and explicit Euler for linear advection, both the data files need to be generated. This means that first one needs to run the *main.py* script setting *preserve_norm=True* and then with it set to *False*.

##### get_data.py

- Purpose : This script downloads the dataset from Zenodo. To do so, the package *zenodo_get* is necessary. An attempt to import it is done, if not available it is asked to install it. The datasets are all saved in the folder *data* and it is checked if they are already avaialable before downloading them. The first run takes a little time, while then this script only verifies the dataset is already available.

##### training.py

- Method : train_network
  
  - Inputs:
    
    - *model* : neural network to train.
    
    - *lr* : the starting learning rate of the optimizer.
    
    - *epochs* : number of training epochs.
    
    - *trainloader* : dataloader containing the test input and time updates.
    
    - *timesteps* : how many time steps to use in the recurrent training.

    - *is_cyclic* : Boolean variable saying if the learning rate scheduler is cyclic, True, or of the step type, False.

    - *is_noise* : Boolean variable saying if the training involves noise injection, True, or not, False.

    - *device* : 'cuda', 'mps', or 'cpu', the device where the operations are performed and the data is stored.
  
  - Purpose : This method implements the training routine for the network *model*. The training loop is quite standard but there are recurrent steps, and there is a progressive pre-training procedure that gradually increases the length of the training sequences while decreasing the learning rate.

##### utils.py

- Class : relu_squared
  
  - Implementation of the activation function $\max(0,x)^2$ based on *nn.Module*.

## Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7665159.svg)](https://doi.org/10.5281/zenodo.7665159)

The data used in this repository can be found on [Zenodo](https://doi.org/10.5281/zenodo.7665159). For completeness, we include the data generation routine in the subfolder `data_generation`. To read more about the data generation procedure [click here](data_generation/README.md).