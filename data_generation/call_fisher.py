'''
call fisher with various functions
'''

#global
import argparse
import pickle
import random
import os
import random as random
from firedrake import sin, pi
#local
import fisher

#Generate a 'random' function
def rand_func(x,y):
    sigma = random.gauss(1,0.5)
    k = random.randint(2,7)
    
    return sigma * sin(k * pi * (x-1)) * sin(k * pi * (y-1))


if __name__=="__main__":
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    args, _ = parser.parse_known_args()

    #Generate data
    t, u = fisher.fisher(rand_func)

    #Save to tmp file
    filename = 'tmp/fisher%s.pickle' % args.iteration
    file = open(filename, 'wb')
    pickle.dump(u,file)
    file.close()
