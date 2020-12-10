import os
import numpy as np
import pickle

from mxlMsleIntra import estimate
from simstudy import rmse

np.random.seed(4711)

###
#Obtain task
###

task = int(os.getenv('TASK')) - 1

filename = "taskplan"
infile = open(filename, 'rb')
taskplan = pickle.load(infile)
infile.close()

S = taskplan[task, 0]
N = taskplan[task, 1]
T = taskplan[task, 2]
r = taskplan[task, 3]

"""
S = 1
N = 1000
T = 5
r = 0
"""

###
#Load data
###

filename = 'sim_S{}_N{}_T{}_r{}'.format(S,N,T,r)
infile = open(filename, 'rb')
sim = pickle.load(infile)
infile.close()

locals().update(sim) 

###
#Estimate MXL via MSLE
###

xFix = np.zeros((0,0)) #np.stack((const2, const3), axis = 1)
xRndUc = np.zeros((0,0)) # #-np.hstack((cost, he, tt))
xRndCo = np.zeros((0,0))

xRnd2Uc = np.zeros((0,0))
xRnd2Co = np.array(xRnd)

#Fixed parameter distributions
#0: normal
#1: log-normal (to assure that fixed parameter is striclty negative or positive)
xFix_trans = np.array([0, 0, 0, 0])

#Random parameter distributions
#0: normal
#1: log-normal
#2: S_B
xRndUc_trans = np.array([0, 0])
xRndCo_trans = np.array([0, 0])

xRnd2Uc_trans = np.array([0, 0])
xRnd2Co_trans = np.array([0, 0])

paramFix_inits = np.zeros((xFix.shape[1],))

paramRndUc_mu_inits = np.zeros((xRndUc.shape[1],))
paramRndUc_sd_inits = np.ones((xRndUc.shape[1],))
paramRndCo_mu_inits = np.zeros((xRndCo.shape[1],))
paramRndCo_ch_inits = 0.1 * np.eye(xRndCo.shape[1])

paramRnd2Uc_mu_inits = np.zeros((xRnd2Uc.shape[1],))
paramRnd2Uc_sdB_inits = np.ones((xRnd2Uc.shape[1],))
paramRnd2Uc_sdW_inits = np.ones((xRnd2Uc.shape[1],))
paramRnd2Co_mu_inits = np.zeros((xRnd2Co.shape[1],))
paramRnd2Co_chB_inits = 0.1 * np.eye(xRnd2Co.shape[1])
paramRnd2Co_chW_inits = 0.1 * np.eye(xRnd2Co.shape[1])

drawsType = 'haltonShiftShuffle'

nDrawsB = 250
nTakesB = 1
nDrawsW = 250
nTakesW = 250
K = 10

seed = 4711

modelName = filename
deleteDraws = True

results = estimate(
        drawsType, nDrawsB, nTakesB, nDrawsW, nTakesW, K,
        seed, modelName, deleteDraws,
        paramFix_inits, 
        paramRndUc_mu_inits, paramRndUc_sd_inits, 
        paramRndCo_mu_inits, paramRndCo_ch_inits,
        paramRnd2Uc_mu_inits, paramRnd2Uc_sdB_inits, paramRnd2Uc_sdW_inits, 
        paramRnd2Co_mu_inits, paramRnd2Co_chB_inits, paramRnd2Co_chW_inits,
        indID, obsID, altID, chosen,
        xFix, xRndUc, xRndCo, xRnd2Uc, xRnd2Co,
        xFix_trans, xRndUc_trans, xRndCo_trans, xRnd2Uc_trans, xRnd2Co_trans) 

###
#Evaluate results
###

res = {}

res['time'] = results['estimation_time']

###
#Parameter recovery
###

zetaMu = results['paramRnd2Co_mu_est']
res['rmse_zetaMu'] = rmse(zetaMu, betaRndInd_true.mean(axis = 0))

tril_idx = np.tril_indices(nRnd)
chB = results['paramRnd2Co_chB_est']
chW = results['paramRnd2Co_chW_est']
SigmaB = chB @ chB.T
SigmaW = chW @ chW.T
res['rmse_SigmaB'] = rmse(SigmaB[tril_idx], np.cov(betaRndInd_true, rowvar = False)[tril_idx])
res['rmse_SigmaW'] = rmse(SigmaW[tril_idx], np.cov(betaRndObs_true, rowvar = False)[tril_idx])

###
#Save results
###

resList = [res, results]

filename = 'results_msle_intra_sim_S{}_N{}_T{}_r{}'.format(S,N,T,r)
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)
outfile.close()