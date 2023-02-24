'''
Solves linear advection with 'conservative' scheme for a given initial
condition function
'''
from firedrake import *
import numpy as np
import matplotlib.pylab as plt

#A test initial condition
def f(x,y):
    return sin(8 * pi * x) * (y-1) * y

#Visualise a firedrake vector as a matrix, only works if structure is
#exactly the same in x and y direction. Implementation hacky AF for time.
def matrify(fd,para):
    m = para.mesh1d
    V = FunctionSpace(m,'CG',degree=para.degree)
    W = VectorFunctionSpace(m,V.ufl_element())
    X = interpolate(m.coordinates, W)
    x = X.dat.data
    y = x #Same size structure already assumed

    mat = np.zeros((para.M,para.M))
    eps = 1e-6
    for i in range(para.M):
        for j in range(para.M):
            x_p = np.remainder(x[i]+eps, 1)
            y_p = np.remainder(y[j]+eps, 1)
            mat[i,j] = fd((x_p,y_p))

    return mat
    


#Define global parameters here
class parameters:
    def __init__(self):
        self.N = 50 #Temporal resolution
        self.M = 100 #Spatial resolution
        self.degree = 1 #Spatial degree
        self.T = 1 #End time
        self.dt = self.T/self.N #time step (should not be changed
                                #independently of N and T)
        self.c = as_vector((1,1)) #Advection vector (assumed fixed globally for
                       #now)

#Main solve routine (takes initial condition function and parameter
#class)
def linadv(ic=f,para=parameters()):

    mesh = PeriodicUnitSquareMesh(para.M,para.M,quadrilateral=True)
    para.mesh1d = PeriodicUnitIntervalMesh(para.M)
    U = FunctionSpace(mesh,'CG',para.degree)

    #Set up initial condition
    u0 = Function(U)
    x, y = SpatialCoordinate(U.mesh())
    u0.interpolate(ic(x,y))

    # #Plot ics (for debugging)
    # plt.matshow(matrify(u0,para))
    # plt.savefig('test.pdf')

    #Build forms
    phi =  TestFunction(U)
    u_trial = TrialFunction(U)
    
    ut = (u_trial-u0)/para.dt #discrete derivative
    u = 0.5*(u_trial+u0) #evaluation point

    F = (ut + inner(para.c, grad(u))) * phi * dx

    #Build solver
    u1 = Function(U)
    prob = LinearVariationalProblem(lhs(F),rhs(F),u1)
    solver = LinearVariationalSolver(prob,
                                     solver_parameters={'mat_type': 'aij',
                                                        'ksp_type': 'preonly',
                                                        'pc_type': 'lu'})
    
    #Set up output
    t = 0
    sol = [matrify(u0,para)]
    time = [t]
    
    #Loop over time and solve
    while (t < para.T-0.5*para.dt):
        t+= para.dt
        solver.solve()
        sol.append(matrify(u1,para))
        time.append(t)

        u0.assign(u1)
        

    return time, sol




if __name__=="__main__":
    t, u = linadv()
    print(t)
    print(len(t))

    for i in range(len(t)):
        plt.matshow(u[i])
        plt.savefig('u%d.pdf' % i)
        plt.close()
