import os
import warnings

from network_linadv import network as network_linadv
from network_heat import network as network_heat
from network_fisher import network as network_fisher

#This method calls the network models for the respective PDEs
def get_network(pde_name='linadv',preserve_norm=None,device='cpu'):
    
    #This check is because we have implemented the projected Euler method
    #only for linear advection. Thus preserve_norm can be True only in that case
    if pde_name!='linadv':
        if preserve_norm!=None:
            warnings.warn('Projected Euler has been implemented only for linadv problem')
            preserve_norm = None
    
    #Instantiation of the correct network, based on the chosen PDE and training regime
    if pde_name=='linadv':
        model = network_linadv(preserve_norm=preserve_norm)
        model.to(device);
    elif pde_name=='heat':
        model = network_heat()
        model.to(device);
    else:
        model = network_fisher()
        model.to(device);
    return model