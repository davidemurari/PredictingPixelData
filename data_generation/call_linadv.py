'''
call linadv with various functions
'''

#global
import argparse
import pickle
import random
import os
from firedrake import sin, cos, pi, exp, Constant

#local
import linadv

#Generate a 'random' function
def rand_func(x,y):
    a = random.randint(1,10)
    b = random.randint(1,10)
    c = random.random()
    e = random.random()
    m1 = random.random()
    m2 = random.random()
    m3 = random.random()
    m4 = random.random()

    x_p = m1 * x + m2 * y
    y_p = m3 * x + m4 * y
    x_ = x_p
    y_ = y_p
    
    return exp(-(x_-e)**2-(y_-c)**2)*sin(2*pi*a*x_)*sin(2*pi*b*y_)*x*(1-x)*y*(1-y) / 0.04


if __name__=="__main__":
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    args, _ = parser.parse_known_args()

    #Generate data
    t, u = linadv.linadv(rand_func)

    #Save to tmp file
    filename = 'tmp/linadv%s.pickle' % args.iteration
    file = open(filename, 'wb')
    pickle.dump(u,file)
    file.close()
