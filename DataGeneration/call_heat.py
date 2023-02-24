'''
call linadv with various functions
'''

#global
import argparse
import pickle
import random
import os
import numpy.random as random
from firedrake import sin, pi, exp
#import numpy.clip as clip
#local
import heat

#Generate a 'random' function
def rand_func(x,y):
    k = random.randint(2,7)
    a = random.normal(1,0.5)
    #a = clip(a,-2,2)
    return a * sin(k*pi*(x-1)) * sin(k*pi*(y-1))


if __name__=="__main__":
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    args, _ = parser.parse_known_args()

    #Generate data
    t, u = heat.heat(rand_func)

    #Save to tmp file
    filename = 'tmp/heat%s.pickle' % args.iteration
    file = open(filename, 'wb')
    pickle.dump(u,file)
    file.close()
