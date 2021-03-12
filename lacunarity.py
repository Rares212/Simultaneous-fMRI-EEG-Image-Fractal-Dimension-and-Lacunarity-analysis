# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:25:48 2021

@author: Rares
"""

import os, sys, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def lacunarity3D(img, minBoxSize=2, nSamples=10, plot=False):
    maxBoxSize = int(np.floor(np.log2(np.min(img.shape))))
    scales = np.floor(np.logspace(minBoxSize, maxBoxSize, num = nSamples, base =2 ))
    scales = np.unique(scales).astype(int)
    
    logLacunarities = np.zeros(len(scales))
    lacunarities = np.zeros(len(scales))
    logScales = np.log(scales)
    for i, r in enumerate(scales):
        counts = np.zeros(r * r * r + 1)
        countsRange = np.arange(r * r * r + 1)
        for x in range(img.shape[0] - r + 1):
            for y in range(img.shape[1] - r + 1):
                for z in range(img.shape[2] - r + 1):
                    boxesTouched = np.count_nonzero(img[x : x + r-1, y : y + r-1, z : z + r-1])
                    counts[boxesTouched] += 1
        counts = counts / ((img.shape[0] - r+1) * (img.shape[1] - r+1) * (img.shape[2] - r+1))
        z1 = np.sum(countsRange * counts)
        z2 = abs(np.sum(countsRange**2 * counts))
        lacunarities[i] = z2 / (z1 * z1)
        logLacunarities[i] = np.log(lacunarities[i])
    #predictedLac, rmse = fitAndPredict(logScales, logLacunarities, logScales)
    # coeffs = np.polyfit(logScales, logLacunarities, 1)
    coeffs = np.zeros(len(scales)) 
    predictedLac = np.polyval(coeffs, logScales)
    if (plot):
        plt.plot(logScales, logLacunarities)
        plt.plot(logScales, predictedLac)
        plt.show()
    return logLacunarities, logScales, coeffs[0]

def fitAndPredict(fx, fy, predict):
    func_linear = lambda params,x: params[0]*x+params[1]
    error_func  = lambda params,fx,fy: func_linear(params,fx)-fy
    final_params,success = leastsq(error_func,(1.0,2.0),args=(np.asarray(fx),np.asarray(fy)))
    predict = func_linear(final_params,predict)
    rmse = np.sqrt(np.mean((predict-fy)**2))
    return predict, rmse