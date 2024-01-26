import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import numpy as np
import pandas as pd
import warnings
import os

#Setting the plotting parameters
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['font.size']= 45
matplotlib.rcParams['font.family']= 'ptm' #'Times New Roman

#Show the predicted time evolution of the initial condition X
#for timesteps_test steps of size dt
def generate_gif_predicted(pde_name,model,X,timesteps_test):
    
    if not os.path.exists('saved_plots'):
        os.mkdir('saved_plots')
    
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    def animate(i):
        cax.cla()
        res = X.clone()
        res = (res.unsqueeze(0))
        for j in range(i):
            res = model(res)
        res = res[0].detach().cpu().numpy()
        im = ax.imshow(res[-1], cmap = 'hot')
        fig.colorbar(im, cax=cax)
        ax.set_title('Prediction, Frame {0}'.format(i))

    ani = animation.FuncAnimation(fig, animate, frames=timesteps_test)
    ani.save(f"saved_plots/predicted_{pde_name}.gif", writer='pillow')
      
#Show the true time evolution of the difference between
#the true and predicted time evolutions for timesteps_test steps
#of size dt, starting from the initial condition X
def generate_gif_true(pde_name,X,Y,timesteps_test):
    
    if not os.path.exists('saved_plots'):
        os.mkdir('saved_plots')
    
    plt.rcParams["figure.autolayout"] = True

    Traj = torch.cat((X,Y),dim=0)

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    def animate(i):
        cax.cla()
        im = ax.imshow((Traj[i]).detach().cpu().numpy(), cmap = 'hot')
        fig.colorbar(im,cax=cax)
        ax.set_title('True dynamics, Frame {0}'.format(i))
    ani = animation.FuncAnimation(fig, animate, frames=timesteps_test)
    ani.save(f"saved_plots/true_{pde_name}.gif", writer='pillow')
    
#Show the time evolution of the initial condition X
#for timesteps_test steps of size dt
def generate_gif_error(pde_name,model,X,Y,timesteps_test):
    
    if not os.path.exists('saved_plots'):
        os.mkdir('saved_plots')
    
    dim = X.shape[0]
    
    plt.rcParams["figure.figsize"] = [10,10]
    plt.rcParams["figure.autolayout"] = True
    
    fig = plt.figure()
    Traj = torch.cat((X,Y),dim=0)
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    def animate(i):
        cax.cla()
        res = X.clone()
        res = (res.unsqueeze(0))
        for j in range(i):
            res = model(res)
        res = res[0,0]
        im = ax.imshow((res-Traj[i]).detach().cpu().numpy(), cmap='hot')
        fig.colorbar(im, cax=cax)
        ax.set_title('Difference of matrices, Frame {0}'.format(i))

    ani = animation.FuncAnimation(fig, animate, frames=timesteps_test)
    ani.save(f"saved_plots/error_{pde_name}.gif", writer='pillow')
    
#Generate .csv files where the three test metrics are stored
#We measure the mean squared error, the  relative error and
#the maximum absolute error for 30 test initial conditions and
#store in these files the mean of such values
def save_test_results(pde_name,model,testloader,preserve_norm=None,is_noise=None):
    
    if not os.path.exists('saved_test_results'):
        os.mkdir('saved_test_results')
    
    if pde_name!='linadv':
        if preserve_norm!=None:
            warnings.warn('Projected Euler has been implemented only for linadv problem')
            preserve_norm = None
    
    model.to('cpu');
    X,Y = next(iter(testloader))
    X,Y = (X.to('cpu')), (Y.to('cpu'))
    Traj = torch.cat((X,Y),dim=1)
    res = X
    
    #Initialize the lists where we store the values
    mseList = []
    max_errorList = []
    relative_error_list = []
    
    mseList.append(torch.mean((res-X)**2).item())
    max_errorList.append(torch.mean(torch.max((torch.abs(res-X)).reshape(len(X),-1),dim=1)[0], dim=0).item())
    relative_error_list.append(torch.mean((torch.linalg.norm((res-X).view(len(X),-1),dim=1,ord=2)  / torch.linalg.norm((X).view(len(X),-1),dim=1,ord=2))).item())
    
    res = model(res)
    
    #Compute the quantities for the successive iterations
    for j in np.arange(1,40):
        mseList.append(torch.mean((res-Traj[:,j:j+1])**2).item())
        max_errorList.append(torch.mean(torch.max((torch.abs(res-Traj[:,j:j+1])).reshape(len(X),-1),dim=1)[0], dim=0).item())
        relative_error_list.append(torch.mean((torch.linalg.norm((res-Traj[:,j:j+1]).view(len(X),-1),dim=1,ord=2)  / torch.linalg.norm((Traj[:,j:j+1]).view(len(X),-1),dim=1,ord=2))).item())
        res = model(res)
    
    noise_tag = "Noise" if is_noise else "NoNoise"
    if pde_name=="linadv":
        pde_tag = "Linadv"
    elif pde_name=="heat":
        pde_tag = "Heat"
    else:
        pde_tag = "Fisher"
    
    #Save the results
    if pde_name == 'linadv':
                
        if preserve_norm==True:
                        
            np.savetxt(f"saved_test_results/{pde_tag}AverageMSEPreserve{noise_tag}.csv", mseList, delimiter=",")
            np.savetxt(f"saved_test_results/{pde_tag}MaxErrorPreserve{noise_tag}.csv", max_errorList, delimiter=",")
            np.savetxt(f"saved_test_results/{pde_tag}RelativeErrorPreserve{noise_tag}.csv", relative_error_list, delimiter=",")
            
        else:
            
            np.savetxt(f"saved_test_results/{pde_tag}AverageMSENoPreserve{noise_tag}.csv", mseList, delimiter=",")
            np.savetxt(f"saved_test_results/{pde_tag}MaxErrorNoPreserve{noise_tag}.csv", max_errorList, delimiter=",")
            np.savetxt(f"saved_test_results/{pde_tag}RelativeErrorNoPreserve{noise_tag}.csv", relative_error_list, delimiter=",")
    else:
        np.savetxt(f"saved_test_results/{pde_tag}AverageMSE{noise_tag}.csv", mseList, delimiter=",")
        np.savetxt(f"saved_test_results/{pde_tag}MaxError{noise_tag}.csv", max_errorList, delimiter=",")
        np.savetxt(f"saved_test_results/{pde_tag}RelativeError{noise_tag}.csv", relative_error_list, delimiter=",")
    print("Results saved or updated in the directory 'saved_test_results'")
    
    

#Generate the error plots from the .csv files
def generate_error_plots(pde_name,model,testloader,preserve_norm=None,is_noise=None):
    
    #Generate the results
    save_test_results(pde_name,model,testloader,preserve_norm,is_noise)
    
    #List of plots we show
    namePlots = ["MaxError", "AverageMSE", "RelativeError"]
    labels = [r"$\texttt{maxE}(j)$", r"$\texttt{mse}(j)$", r"$\texttt{rE}(j)$"]
    
    noise_tag = "Noise" if is_noise else "NoNoise"
    if pde_name=="linadv":
        pde_tag = "Linadv"
    elif pde_name=="heat":
        pde_tag = "Heat"
    else:
        pde_tag = "Fisher"
    
    if not os.path.exists('saved_plots'):
        os.mkdir('saved_plots')
    
    #Generation of the plots for the 3 different PDEs
    if pde_name=="linadv":
        for it, name in enumerate(namePlots):
            is_data_constr = False
            is_data_unconstr = False
            
            try:
                dfConstr = pd.read_csv(f'saved_test_results/{pde_tag}{name}Preserve{noise_tag}.csv',header=None)
                is_data_constr = True
            except:
                print(f"Missing data for the {name} of the conserved case")
            try:
                dfUnconstr = pd.read_csv(f'saved_test_results/{pde_tag}{name}NoPreserve{noise_tag}.csv',header=None)
                is_data_unconstr = True
            except:
                print(f"Missing data for the {name} of the non conserved case")
                
            xx = np.arange(0,40)
            if is_data_unconstr or is_data_constr:
                fig = plt.figure(figsize=[10,10],dpi=300)
            if is_data_constr:
                plt.plot(xx,dfConstr.iloc[:len(xx),-1],'r-o',label="Constrained")
            if is_data_unconstr:
                plt.plot(xx,dfUnconstr.iloc[:len(xx),-1],'b-o',label="Unconstrained")
            
            if is_data_constr==False and is_data_unconstr==False:
                print(f"There is nothing saved to generate the plots of {name}")
            else:
                plt.yticks(fontsize=45)
                plt.xticks(fontsize=45)
                plt.legend(fontsize=45)
                plt.xlabel("Number of iterations",fontsize=45)
                plt.ylabel(f"{labels[it]}",fontsize=45)
                if is_data_unconstr:
                    ymax = np.max(dfUnconstr.iloc[:,-1])
                else:
                    ymax = np.max(dfConstr.iloc[:,-1])
                ymin = 0
                ylist = np.linspace(ymin,ymax,4)
                plt.yticks(ylist)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(f"saved_plots/{pde_tag}{name}{noise_tag}.pdf", format="pdf",bbox_inches='tight')                
    else:
        for it, name in enumerate(namePlots):
            is_data = False
            
            try:
                df = pd.read_csv(f'saved_test_results/{pde_tag}{name}{noise_tag}.csv',header=None)
                is_data = True
            except:
                print(f"Missing data for the {name} of {pde_name}, noise {noise_tag}")
                
            xx = np.arange(0,40)
            if is_data:
                fig = plt.figure(figsize=[10,10],dpi=300)
                plt.plot(xx,df.iloc[:len(xx),-1],'r-o')
                plt.yticks(fontsize=45)
                plt.xticks(fontsize=45)
                plt.xlabel("Number of iterations",fontsize=45)
                plt.ylabel(f"{labels[it]}",fontsize=45)
                ymax = np.max(df.iloc[:,-1])
                ymin = 0
                ylist = np.linspace(ymin,ymax,4)
                plt.yticks(ylist)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(f"saved_plots/{pde_tag}{name}{noise_tag}.pdf", format="pdf",bbox_inches='tight')
            else:
                print(f"There is nothing saved to generate the plots of {name} in {pde_name}")
    print("Plots saved or updated in the directory 'saved plots'")