import os
#import sys
import numpy as np
import pickle

###
#Set seed
###

np.random.seed(4711)

###
#Generate plan of tasks
###

S = np.array([1, 2]).reshape((-1,1))
N = np.array([1000]).reshape((-1,1))
T = np.array([10, 20]).reshape((-1,1))
R = 20

N_plan = np.kron(N, np.ones((T.shape[0], 1)))
T_plan = np.kron(np.ones((N.shape[0], 1)), T)
NT_plan = np.kron(np.hstack((N_plan, T_plan)), np.ones((R , 1)))
R_plan = np.kron(np.ones((N.shape[0] * T.shape[0], 1)), np.arange(R).reshape((R, 1)))
NTR_plan = np.hstack((NT_plan, R_plan))
S_plan = np.kron(S, np.ones((NTR_plan.shape[0], 1)))
SNTR_plan = np.hstack((S_plan, np.kron(np.ones((S.shape[0], 1)), NTR_plan)))

taskplan = np.array(SNTR_plan, dtype = 'int64')

###
#Save
###

filename = "taskplan"
if os.path.exists(filename): os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(taskplan, outfile)
outfile.close()