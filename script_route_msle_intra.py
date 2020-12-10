import os
import numpy as np
import pickle

from mxlMsleIntra import estimate
from simstudy import rmse

np.random.seed(4711)

###
#Load data
###

r = int(os.getenv('TASK')) - 1

filename = 'data_route_r{}'.format(r)
infile = open(filename, 'rb')
data = pickle.load(infile)
infile.close()

locals().update(data)

###
#Estimate MXL via MSLE
###

#xFix = np.zeros((0,0)) #np.stack((const2, const3), axis = 1)
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

drawsType = 'mlhs'

nDrawsB = 500
nTakesB = 1
nDrawsW = 500
nTakesW = 500
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
#Save results
###

resList = [res, results]

filename = 'results_route_msle_intra_r{}'.format(r)
if os.path.exists(filename): 
    os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)
outfile.close()