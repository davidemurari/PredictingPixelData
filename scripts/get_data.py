import os

def download_data(pde_name):

    name = f"data_{pde_name}.pickle" if pde_name=="linadv" \
        else f"data_{pde_name}_periodic.pickle"
    name_verification = f"data_{pde_name}_verification.pickle" if pde_name=="linadv" \
        else f"data_{pde_name}_periodic_verification.pickle"

    #Downloading and saving the dataset into the data directory
    if not os.path.exists('data'):
        
        try:
            import zenodo_get
        except:
            input('To download the data the package ''zenodo_get'' needs to be imported.\n Press enter to agree on dowloading it.') #if you press enter you go on
            os.system('pip install zenodo_get')
        
        working_directory = os.getcwd()
        os.mkdir('data')
        os.chdir('data')
        os.system(f'zenodo_get 11549488') #for the version 2 of the dataset
        os.chdir(working_directory)
    else:
        working_directory = os.getcwd()
        os.chdir('data')
        if not os.path.exists(name) or not os.path.exists(name_verification):
            try:
                import zenodo_get
            except:
                input('To download the data the package ''zenodo_get'' needs to be imported.\n Press enter to agree on dowloading it.') #if you press enter you go on
                os.system('pip install zenodo_get')
            
            os.system(f'zenodo_get 11549488') #for the version 2 of the dataset
            
        os.chdir(working_directory)