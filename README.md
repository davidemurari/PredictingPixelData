# Predictions based on pixel data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7678902.svg)](https://doi.org/10.5281/zenodo.7678902)

This repository is a suppliment to the manuscript **Predictions based on pixel data: Insights from PDEs and finite differences** by _Elena Celledoni, James Jackaman, Davide Murari and Brynjulf Owren_. 

Necessary installed libraries with the version tested for the implementation:

- Matplotlib v. 3.7.1
- Numpy v. 1.22.4
- PyTorch v. 2.0.0
- Pandas v. 1.4.4

If not already available, they can all be installed via pip as

> pip install libraryName

The implementation includes methods that allow to work with the three following PDEs:

- Linear advection : $\partial_t u = \boldsymbol{b}\cdot \nabla u = \partial_xu+\partial_yu$

- Heat equation : $\partial_t u = \alpha \Delta u=\alpha(\partial_{xx}u+\partial_{yy}u)$

- Fisher equation : $\partial_t u = \alpha \Delta u + u(1-u)$

The main file *main.py* allows to interactively choose which of the three is of interest and the specifics of the simulation.

> **Quickstart:** To run the code as set, run the command
> 
>     python main.py
> 
> and then the user is guided through the choices. An example run is as follows:
> 
> - *Enter the pde you want (choose among 'linadv', 'heat', 'fisher'):* heat
> 
> - *Enter how many timesteps you want in the training data (1<=t<=5):* 5
> 
> - *Do you want to train a new model or use a pre-trained one? ('train','pretrained'):* pretrained
> 
> - *Write yes to plot the errors:* yes
> 
> This allows to generate the error plots included in the paper for the heat equation.

##### main.py

After importing the necessary Python libraries, the main script asks the user to set the following parameters:

- *pde_name* : name of the PDE to work with among 'linadv', 'heat' and 'fisher'

- *conserve_norm* : this is set to None for heat and fisher equations, while it can be set to True/False for the linear advection one. If True, projection Euler method is used in the training and the network will be norm-preserving. If False, explicit Euler is the method of choice.

- *timesteps* : determines how many steps (integer value from 1 to 5) to use in the (recurrent) training procedure

- *train* : this is a Boolean variable which if set to True will call the training procedure for the network, otherwise a pre-trained model will be loaded in order to show the results

Then the training and test sets are generated and the neural network with the chosen parameters is instantiated.

The training parameters one can choose are the learning rate, the training epochs, the batch size and the weight decay parameter. They can be set manually from this script. The optimizer and the learning rate scheduling can be replaced in the *train* method contained in the scripts *methods_linadv.py*, *methods_heat.py*, *methods_fisher.py*.

If 'cuda' device is available, all the calculations and the storage relies on the gpu. Otherwise the cpu is used.

After either training or loading the pre-trained networks, it is possible to display the results with the implemented plotting tools. See the script *generate_plots.py* for more details on them.

---

We now describe the scripts included in the repository that are called by the *main.py* script.

##### create_dataset.py

- Class : dataset
  
  - Parameters :
    
    - *x,y* : initial condition of the PDE, time updates
    
    - *device* : determines on which device will the object of this class live, usually 'cpu' or 'cuda'
  
  - Purpose :
    
    - This class defines the train and test dataloaders

- Method : get_train_test_split
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied
    
    - *timesteps* : decides how many discrete steps are included in the training and test sets
    
    - *device* : determines the device on which the train and test loader live
  
  - Purpose : This method returns the train and test loader obtained by loading the data points from zenodo (see script *get_data.py*) and splitting them into training and test sets. These datasets are then returned as dataloaders thanks to the class *dataset*.

##### generate_network.py

- Method : get_network
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied
    
    - *preserve_norm*: a Boolean varialbe which is None for heat and Fisher equation by default, and can be True/False for the advection equation. True if projected Euler is used to train the network, False if explicit Euler.
    
    - *device* : determines the device on which the generated network will be loaded
  
  - Purpose : This method aims to instantiate the neural network that will be then trained by the *train.py* script. The output is the object *model* that is loaded on the *device*. This method calls other methods in the scripts methods_fisher.py, methods_heat.py, methods_linadv.py, that are specific to the problem of interest. A warning is displayed if *preserve_norm* is not None for the heat equation and Fisher equation.

##### generate_plots.py

- Method : generate_gif_predicted
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied
    
    - *model* : trained neural network to adopt in order to make predictions
    
    - *X* : test initial condition for the plot
    
    - *timesteps_test* : number of performed steps to generate the output .gif file
  
  - Purpose : This method generates a .gif file showing the time evolution of the initial condition *X* updated by *model* for *timesteps_test* iterations. The result is saved in the folder *savedPlots* as *predicted_{pde_name}.gif*

- Method : generate_gif_true
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied
    
    - *X* : test initial condition for the plot
    
    - *Y* : true time evolution of the initial condition *X* given by the PDE discrete flow map
    
    - *timesteps_test* : number of performed steps to generate the output .gif file
  
  - Purpose : This method generates a .gif file showing the time evolution of the initial condition *X* updated by a numerical method solving the PDE with initial condition *X* and for *timesteps_test* steps. The result is saved in the folder *savedPlots* as *true_{pde_name}.gif*

- Method : generate_gif_error
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied
    
    - *X* : test initial condition for the plot
    
    - *model* : trained neural network to adopt in order to make predictions
    
    - *Y* : true time evolution of the initial condition *X* given by the PDE discrete flow map
    
    - *timesteps_test* : number of performed steps to generate the output .gif file
  
  - Purpose : This method generates a .gif file showing the time evolution of the difference between the predictions of *model* and the exact update stored in *Y*. The result is saved in the folder *savedPlots* as *error_{pde_name}.gif*

- Method : save_test_results
  
  - Inputs :
    
    - *pde_name* : a string among 'linadv', 'heat' and 'fisher' determining which PDE is studied
    
    - *model* : trained neural network to adopt in order to make predictions
    
    - *testloader* : dataloader for the test set used to compute the error metrics
    
    - *preserve_norm* : Boolean variable which is by default set to None for heat and Fisher equation, and can be True/False for the advection equation.
  
  - Purpose : Generate .csv files where the three test metrics are stored. We measure the mean squared error, the relative error and the maximum absolute error for 30 test initial conditions and store in these files the mean of such values. The files are saved in the folder *saved_test_results*.

- Method : generate_error_plots
  
  - Same inputs as *save_test_results*
  
  - Purpose : After generating the .csv files with the method *save_test_results*, this method displays and saves the correspondent plots. These plots are saved as .pdf files in the folder *saved_plots*. The method plots data available in the folder *saved_test_results*. Thus, to display the comparison between projected Euler and explicit Euler for linear advection, both the data files need to be generated. This means that first one needs to run the *main.py* script setting *preserve_norm=True* and then with it set to *False*.

##### get_data.py

- Purpose : This script downloads the dataset from Zenodo. To do so, the package *zenodo_get* is necessary. An attempt to import it is done, if not available it is asked to install it. The datasets are all saved in the folder *data* and it is checked if they are already avaialable before downloading them. The first run takes a little time, while then this script only verifies the dataset is already available.

##### network_fisher.py, network_heat.py, network_linadv.py

We group together these three scripts because they all have the same structure. What changes is mostly the network architecture.

- Class : network
  
  - Parameters :
    
    - No parameter for the heat and Fisher scripts
    
    - *preserve_norm* : Boolean variable which if true makes the forward pass be based on projected Euler method. If False the explicit Euler method is used instead.
  
  - Purpose : This method aims to define the neural network architecture. The weights are divided into weights related to the $\mathrm{ReLU}$ activation function, and to the $\mathrm{ReLU}^2$ activation function. They are initialized and then in the *F* method the approximate vector field is defined as a feed-forward neural network involving all the aforementioned weights. The forward pass performs *nlayers* steps of size *dt/nlayers*.

##### training.py

- Method : train
  
  - Inputs:
    
    - *model* : neural network to train
    
    - *lr* : the starting learning rate of the optimizer
    
    - *weight_decay* : weight decay parameter for the optimizer
    
    - *epochs* : number of training epochs
    
    - *trainloader* : dataloader containing the test input and time updates
    
    - *timesteps* : how many time steps to use in the recurrent training
  
  - Purpose : This method implements the training routine for the network *model*. The training loop is quite standard but there are recurrent steps, and there is a progressive pre-training procedure that gradually increases the length of the training sequences while decreasing the learning rate.

##### utils.py

- Class : relu_squared
  
  - Implementation of the activation function $\max\{0,x\}^2$ based on *nn.Module*.

## Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7665159.svg)](https://doi.org/10.5281/zenodo.7665159)

The data used in this repository can be found on [Zenodo](https://doi.org/10.5281/zenodo.7665159). For completeness, we include the data generation routine in the subfolder `DataGeneration`. To read more about the data generation procedure [click here](DataGeneration/README.md).