import os
#import sys
import numpy as np
import pickle

from mxlMcmc import estimate, predictB, predictB_plugin, predictW, predictW_plugin
from simstudy import quad_loss

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
#Estimate MNL via MCMC
###

xFix = np.hstack((xRnd, xFix))
xRnd = np.zeros((0,0))

#Fixed parameter distributions
#0: normal
#1: log-normal (to assure that fixed parameter is striclty negative or positive)
xFix_trans = np.array([0, 0])

#Random parameter distributions
#0: normal
#1: log-normal
#2: S_B
xRnd_trans = np.array([0, 0])

paramFix_inits = np.zeros((xFix.shape[1],))
zeta_inits = np.zeros((xRnd.shape[1],))
Omega_inits = 0.1 * np.eye(xRnd.shape[1])

A = 1e3
nu = 2
diagCov = False

mcmc_nChain = 2
mcmc_iterBurn = 50000
mcmc_iterSample = 50000
mcmc_thin = 5
mcmc_iterMem = mcmc_iterSample
mcmc_disp = 1000
seed = 4711
simDraws = 100  

rho = 0.1
rhoF = 0.01

modelName = 'd_mcmc_flat_' + filename
deleteDraws = True

results = estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simDraws,
        rhoF, rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        paramFix_inits, zeta_inits, Omega_inits,
        indID, obsID, altID, chosen,
        xFix, xRnd,
        xFix_trans, xRnd_trans)


###
#Prediction: Between
###

nTakes = 10
nSim = 1000

mcmc_thinPred = 1
mcmc_disp = 1000
deleteDraws = False

xFix_valB = np.hstack((xRnd_valB, xFix_valB))
xRnd_valB = np.zeros((0,0))

"""
pPredB = predictB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID_valB, obsID_valB, chosen_valB,
        xFix_valB, xRnd_valB)
"""

zeta = results['postMean_zeta']
Omega = results['postMean_Omega']
paramFix = results['postMean_paramFix']
pPredB_plugin = predictB_plugin(
        zeta, Omega, paramFix,
        nTakes, nSim,
        indID_valB, obsID_valB, chosen_valB,
        xFix_valB, xRnd_valB)

###
#Prediction: Within
###

mcmc_thinPred = 1
mcmc_disp = 1000
deleteDraws = True

xFix_valW = np.hstack((xRnd_valW, xFix_valW))
xRnd_valW = np.zeros((0,0))

"""
pPredW = predictW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp,
        seed,
        modelName, deleteDraws,
        indID_valW, obsID_valW, chosen_valW,
        xFix_valW, xRnd_valW)
"""

paramRnd = results['postMean_paramRnd']
pPredW_plugin = predictW_plugin(
        paramFix, paramRnd,
        indID_valW, obsID_valW, chosen_valW,
        xFix_valW, xRnd_valW)

###
#Evaluate results
###

res = {}

res['time'] = results['estimation_time']
res['probB_chosen_plugin'] = np.mean(pPredB_plugin[chosen_valB == 1])
res['brierB_plugin'] = np.mean((pPredB_plugin - chosen_valB)**2)
res['probW_chosen_plugin'] = np.mean(pPredW_plugin[chosen_valW == 1])
res['brierW_plugin'] = np.mean((pPredW_plugin - chosen_valW)**2)

print(res)

###
#Save results
###

resList = [res, results]

filename = 'results_route_mcmc_flat_r{}'.format(r)
if os.path.exists(filename): 
    os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)