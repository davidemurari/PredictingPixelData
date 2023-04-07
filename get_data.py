import os

#Import the zenodo_get package to download the dataset
try:
    import zenodo_get
except:
    input('To download the data the package ''zenodo_get'' needs to be imported.\n Press enter to agree on dowloading it.') #if you press enter you go on
    os.system('pip install zenodo_get')
    

#Downloading and saving the dataset into the data directory
if not os.path.exists('data'):
    working_directory = os.getcwd()
    os.mkdir('data')
    os.chdir('data')
    os.system(f'zenodo_get 7665159')
    os.chdir(working_directory)