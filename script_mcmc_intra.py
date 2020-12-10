import os
#import sys
import numpy as np
import pickle

from mxlMcmcIntra import estimate, predictB, predictB_plugin, predictW, predictW_plugin
from simstudy import rmse, quad_loss

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
#Estimate MXL via MCMC
###

#Random parameter distributions
#0: normal
#1: log-normal
#2: S_B
xFix = np.zeros((0,0))
xRnd_trans = np.array([0, 0])

paramFix_inits = np.zeros((xFix.shape[1],))
zeta_inits = np.zeros((xRnd.shape[1],))
OmegaB_inits = 0.1 * np.eye(xRnd.shape[1])
OmegaW_inits = 0.1 * np.eye(xRnd.shape[1])

A = 1e3
nu = 2
diagCov = (False, False)

mcmc_nChain = 2
mcmc_iterBurn = 200000
mcmc_iterSample = 200000
mcmc_thin = 10
mcmc_iterMem = int(mcmc_iterSample / 10)
mcmc_disp = 1000
seed = 4711
simLogLik = False
simLogLikDrawsType = 'mlhs'
simDraws = 200  

rho = 0.1
rhoF = 0.01

modelName = 'd_mcmc_intra_' + filename
deleteDraws = True

results = estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simLogLik, simLogLikDrawsType, simDraws,
        rhoF, rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        paramFix_inits,
        zeta_inits, OmegaB_inits, OmegaW_inits,
        indID, obsID, altID, chosen,
        xFix,
        xRnd, xRnd_trans)

###
#Prediction: Between
###

nTakes = 1
nSim = 200

mcmc_thinPred = 2
mcmc_disp = 1000
deleteDraws = False

xFix_valB = np.zeros((0,0))

"""
pPredB = predictB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID_valB, obsID_valB, altID_valB, chosen_valB,
        xRnd_valB)
"""

nTakes = 1
nSim = 2000
paramFix = None
zeta = results['postMean_zeta']
OmegaB = results['postMean_OmegaB']
OmegaW = results['postMean_OmegaW']
pPredB_plugin = predictB_plugin(
        paramFix,
        zeta, OmegaB, OmegaW, 
        nTakes, nSim,
        indID_valB, obsID_valB, altID_valB, chosen_valB,
        xFix_valB, xRnd_valB)


###
#Prediction: Within
###

nTakes = 10
nSim = 1000

mcmc_thinPred = 2
mcmc_disp = 1000
deleteDraws = True

xFix_valW = np.zeros((0,0))

"""
pPredW = predictW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID_valW, obsID_valW, altID_valW, chosen_valW,
        xRnd_valW)
"""

paramRndB = results['postMean_paramRndB']
pPredW_plugin = predictW_plugin(
        paramFix,
        paramRndB, OmegaW,
        nTakes, nSim,
        indID_valW, obsID_valW, altID_valW, chosen_valW,
        xFix_valW, xRnd_valW)

###
#Evaluate results
###

res = {}

res['time'] = results['estimation_time']
#res['qlB'] = quad_loss(pPredB, pPredB_true)
res['qlB_plugin'] = quad_loss(pPredB_plugin, pPredB_true)
#res['probB_chosen'] = np.mean(pPredB[chosen_valB == 1])
res['probB_chosen_plugin'] = np.mean(pPredB_plugin[chosen_valB == 1])
res['brierB_plugin'] = np.mean((pPredB_plugin - chosen_valB)**2)
#res['qlW'] = quad_loss(pPredW, pPredW_true)
res['qlW_plugin'] = quad_loss(pPredW_plugin, pPredW_true)
#res['probW_chosen'] = np.mean(pPredW[chosen_valW == 1])
res['probW_chosen_plugin'] = np.mean(pPredW_plugin[chosen_valW == 1])
res['brierW_plugin'] = np.mean((pPredW_plugin - chosen_valW)**2)

###
#Parameter recovery
###

zetaMu = results['postMean_zeta']
res['rmse_zetaMu'] = rmse(zetaMu, betaRndInd_true.mean(axis = 0))

tril_idx = np.tril_indices(nRnd)
SigmaB = results['postMean_OmegaB']
SigmaW = results['postMean_OmegaW']
res['rmse_SigmaB'] = rmse(SigmaB[tril_idx], np.cov(betaRndInd_true, rowvar = False)[tril_idx])
res['rmse_SigmaW'] = rmse(SigmaW[tril_idx], np.cov(betaRndObs_true, rowvar = False)[tril_idx])


###
#Save results
###

resList = [res, results]

filename = 'results_mcmc_intra_sim_S{}_N{}_T{}_r{}'.format(S,N,T,r)
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)
outfile.close()