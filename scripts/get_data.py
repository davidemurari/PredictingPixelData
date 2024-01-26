import os

def download_data(pde_name):

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
        os.system(f'zenodo_get 7665159')
        os.chdir(working_directory)
    else:
        working_directory = os.getcwd()
        os.chdir('data')
        if not os.path.exists(f'data_{pde_name}.pickle') or not os.path.exists(f'data_{pde_name}_verification.pickle'):
            try:
                import zenodo_get
            except:
                input('To download the data the package ''zenodo_get'' needs to be imported.\n Press enter to agree on dowloading it.') #if you press enter you go on
                os.system('pip install zenodo_get')
            
            os.system(f'zenodo_get 7665159')
            
        os.chdir(working_directory)