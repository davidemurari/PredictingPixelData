# Predictions based on pixel data

This repository is a suppliment to the manuscript **Predictions based on pixel data: Insights from PDEs and finite differences** by _Elena Celledoni, James Jackaman, Davide Murari and Brynjulf Owren_. 



## Data generation

For completeness, we include the data generation routine used in the associated paper, and all code related to data generation can be found in the subfolder `DataGeneration`. 

### Installation of dependencies

The data generation routine depends on [Firedrake](https://www.firedrakeproject.org).

#### Local installation

Firedrake can be installed locally can be installed locally. The Firedrake install script can be downloaded [here](https://www.firedrakeproject.org/download.html), and can be installed into a new Python virtual envionrment with `python3 firedrake-install`. To install the version used in the paper input `python3 firedrake-install --doi 10.5281/zenodo.7414962`.

#### Working on Google colab

To use Firedrake on Google colab one can utilise [FEM on Colab](https://fem-on-colab.github.io). To use this just include the cell

```
#import finite element library
try:
    import firedrake
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/firedrake-install-real.sh" -O "/tmp/firedrake-install.sh" && bash "/tmp/firedrake-install.sh"
    import firedrake
    
#allow for local imports
from google.colab import drive
drive.mount('/content/gdrive')
%cd "/content/gdrive/My Drive/path/to/DataGeneration"
```

and interact with the python scripts by importing them as modules.

### Problems which can be solved

Data can be generated for three (relatively simple) finite element discretisations of the following evolution PDEs (in two dimensional space):

- Linear advection: $u_t = {\bf c} \cdot \nabla u$ with periodic spatial boundary conditions.
- Heat equation: $u_t = \alpha \Delta u$ with zero dirichlet spatial boundary conditions.
- Fisher equation: $u_t = \alpha \Delta u + u(1-u)$ with zero dirichlet spatial boundary conditions.

### How to run the code

Each problem is comprised of three python scripts, which have two main functions. 

1. Generating large amounts of data.
2. Generating one simulation.

#### Generating large amounts of data

To generate large amounts of data run `generate_problemname.py`. This file executes multiple processes on single cores in parallel, with the number determined by the variable `MaxProcesses` with a default value of `8`. The python script will save the data to `data_problemname.pickle`. The problem and discretisation parameters are set in `problemname.py` in the `parameters` class. The random initial conditions are specified in `call_problemname.py`.

#### Generating one simulation

To run a simulation run `problemname.py`, or import the function `problemname` from `problemname.py`. The function `problemname` takes as input an initial condition and a parameters class. When running a single simulation by calling the python file the solution snapshots will be plotted and saved in the current directory.