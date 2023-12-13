#imports
import pandas as pd
import pickle
import subprocess
import time


MaxProcesses = 40
Processes = []

def checkrunning():
    for p in reversed(range(len(Processes))):
        if Processes[p].poll() is not None:
            del Processes[p]
    return len(Processes)


iterations = 1000 #number of data to generate

#generate
for i in range(iterations):
    print('i =', i, 'initialised')

    process = subprocess.Popen('python call_adv.py --iteration %s' % i,
                               shell=True, stdout=subprocess.PIPE)
    Processes.append(process)

    while checkrunning()==MaxProcesses:
        time.sleep(1)

while checkrunning()!=0:
    time.sleep(1)
    

dict = {}
#load generated data
for i in range(iterations):
    try:
        filename = 'tmp/adv%d.pickle' % i
        with open(filename,'rb') as file:
            u_ = pickle.load(file)
            dict.update({i: u_})
    except Exception as e:
        print('Loading iterate %s failed with' % i)
        print('Error:' +str(e))
        

#Save
filename = 'data_adv.pickle'
file = open(filename,'wb')
pickle.dump(dict,file)
file.close()
