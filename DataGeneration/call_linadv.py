'''
call linadv with various functions
'''

#global
import argparse
import pickle
import random
import os
from firedrake import sin, cos, pi

#local
import linadv

#Generate a 'random' function
def rand_func(x,y):
    shift_x = random.random()
    shift_y = random.random()
    int1 = random.randint(5,8)
    int2 = random.randint(5,8)
    
    return sin(2 * pi * int1 * (x-shift_x)) * cos(2 * pi * int2 * (y-shift_y)) + 1


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
