'''
Nonlinear reaction-diffusion equation solver
'''
from firedrake import *
import numpy as np
import matplotlib.pylab as plt
import numpy.random as random

#A test initial condition
def f(x,y):
    return sin(2*pi*(x-0.2)) * cos(2*pi*(y-0.3))


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
    eps = 0 #no remainder needed on non-periodic mesh
    for i in range(para.M):
        for j in range(para.M):
            x_p = np.remainder(x[i]+eps, 1)
            y_p = np.remainder(y[j]+eps, 1)
            mat[i,j] = fd((x_p,y_p))

    return mat

#Define global parameters here
class parameters:
    def __init__(self):
        self.alpha = .01 #rate of dissipation
        self.M = 100 #Spatial resolution
        dx = 1/self.M
        self.dt = 0.24 * dx**2 / self.alpha
        self.N = 10 #Temporal resolution
        self.degree = 1 #Spatial degree
        self.T = self.N*self.dt #End time
        
#Main solve routine (takes initial condition function and parameter
#class)
def fisher(ic=f,para=parameters()):

    mesh = PeriodicUnitSquareMesh(para.M,para.M,quadrilateral=True)
    para.mesh1d = PeriodicUnitIntervalMesh(para.M)
    U = FunctionSpace(mesh,'CG',para.degree)

    #Set up initial condition
    u0 = Function(U)
    x, y = SpatialCoordinate(U.mesh())
    u0.interpolate(ic(x,y))

    # #Plot ics (for debugging)
    # plt.matshow(matrify(u0,para))
    # plt.show()

    #Build forms
    phi =  TestFunction(U)
    u1 = Function(U)
    u1.assign(u0)
    
    ut = (u1-u0)/para.dt #discrete derivative
    u =  0.5*(u1+u0) #evaluation point

    #u(1-u)
    nl = u - 1/3 * (u1*u1 + u1*u0 + u0*u0)
    
    F = ut * phi * dx \
        + para.alpha * inner(grad(u),grad(phi)) * dx \
        - nl * phi * dx

    #bcs = DirichletBC(U, Constant(0), (1,2,3,4))
    
    #Build solver
    prob = NonlinearVariationalProblem(F,u1)
    solver = NonlinearVariationalSolver(prob,
                                        solver_parameters={#
                                            'ksp_type': 'preonly',
                                            'pc_type': 'lu',
                                            'snes_type': 'newtonls',
                                            'snes_rtol': 1e-10,
                                            'snes_stol': 1e-10,
                                            'snes_monitor': None,
                                        })
    
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
    t, u = fisher()
    print(t)
    print(len(t))

    for i in range(len(t)):
        plt.matshow(u[i])
        plt.colorbar(plt.imshow(u[i]))
        plt.savefig('u%d.pdf' % i)
        plt.close()
