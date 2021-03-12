# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:53:23 2021

@author: Rares
"""

import os
from sys import stdout
import scipy.io
import scipy
import numpy as np

import mloop
from mloop.processing import mrifrac, label, eeg, lacunarity
from mloop.data import reading
import mloop.visual as blExport
from mloop.learning import predict

import nibabel as nib
from nilearn import plotting
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import nilearn.image as nimage
from nilearn.input_data import NiftiMasker

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

from mne import channels, viz, io
import mne

import csv

import xgboost as xgb

import scipy
from scipy import signal, fftpack
from scipy.signal import welch
from scipy.integrate import simps
from scipy import stats

from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd

# from bayes_opt import BayesianOptimization

# import tsfresh as fresh
# from tsfresh import extract_features
# from tsfresh.feature_extraction.settings import MinimalFCParameters
# from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
#     load_robot_execution_failures


scriptsDir = os.path.dirname(os.path.realpath(__file__))
mainDir = os.path.dirname(scriptsDir)
datasetDir = '{}\\Dataset'.format(mainDir)
dataDir = '{}\\Data'.format(mainDir)
photosDir = '{}\\Photos'.format(mainDir)
generatedDir = '{}\\Generated'.format(mainDir)

fmri1TestPath = '{}\\sub-01\\func\\sub-01_task-main_run-01_bold.nii.gz'.format(datasetDir)
fmri1EventsTestPath = '{}\\sub-01\\func\\sub-01_task-main_run-01_events.tsv'.format(datasetDir)
fmri2TestPath = '{}\\sub-01\\func\\\sub-01_task-main_run-01_bold.nii.gz'.format(datasetDir)
fmri2EventsTestPath = '{}\\sub-01\\func\\sub-01_task-main_run-02_events.tsv'.format(datasetDir)
anatTestPath = '{}\\sub-01\\anat\\sub-01_T1map.nii.gz'.format(datasetDir)
eeg1TestPath = '{}\\sub-01\\EEG\\EEG_data_sub-01_run-01.mat'.format(datasetDir)
eeg1EventsTestPath = '{}\\sub-01\\EEG\\EEG_events_sub-01_run-01.mat'.format(datasetDir)
eeg2TestPath = '{}\\sub-01\\EEG\\EEG_data_sub-01_run-02.mat'.format(datasetDir)
eeg2EventsTestPath = '{}\\sub-01\\EEG\\EEG_events_sub-01_run-02.mat'.format(datasetDir)
electrodeLocationsPath = '{}\\additional_files\\electrode_info.txt'.format(datasetDir)

if __name__ == '__main__':

    # FOR 2D SKELETONIZATION: lowThresh = 150, highThresh = 230
    # SLICE TIMING: 2s per slice
    # ELP Kind - BESA format, 64 electrodes
    
    # masker = NiftiMasker(standardize=True, mask_strategy='epi')
    # masker.fit(fmri1TestPath)
    # fmri_masked = masker.fit_transform(fmri1TestPath)
    
    (fmriStartIndex, eegStartIndex) = label.timeAlign(fmri1EventsTestPath, eeg1EventsTestPath)
    
    (eegData, fs, excludedChannels) = reading.readEEG(eeg1TestPath)
    
    fmri = nimage.load_img(fmri1TestPath)
    
    nFrames = fmri.shape[3]
    frameTime = 2
    t = np.arange(0, nFrames * frameTime, frameTime)
    fractalDimensionFrames = np.zeros(nFrames)
    lacunarityFrames = np.zeros(nFrames)
    spectralPowerFrames = np.zeros(nFrames)
    
    # (imgParcels, ward) = label.clusterWard(fmri, standardize=True)
    # imgParcels = imgParcels[fmriStartIndex:]
    
    eegEndIndex = eegStartIndex + nFrames * frameTime * fs
    eegData = eegData[:, eegStartIndex:eegEndIndex]
    #eegData = eegData[:, eegStartIndex:eegStartIndex+4000]
    #eegData = eeg.fillExcludedChannels(eegData, excludedChannels[0, 0])
    nElectrodes = eegData.shape[0]
    
    fs = 1000
    frameSize = 2000
    downsampleFactor = 8
    eegData = scipy.signal.decimate(eegData, downsampleFactor)
    fs = fs / downsampleFactor
    frameSize = int(frameSize / downsampleFactor)
    
    eegDataSplit = eeg.eegSplit(eegData, frameSize)

    thresholds = [12500]

    montage = channels.read_custom_montage(electrodeLocationsPath)
    eegInfo = mne.create_info(montage.ch_names, fs, "eeg")
    raw = io.RawArray(eegData * 0.000001, eegInfo, 0)
    raw.set_montage(montage)
    # raw.filter(1, 40)
    # ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.plot_properties(raw)
    
    
    #print(montage)
    #viz.plot_montage(montage)

    for t in thresholds:
        print("Current Thresh: " + str(t))
        for i in range(0, nFrames):            
            print("Current frame: " + str(i))
            fmriFrame = nimage.index_img(fmri, i)
            #testFrame = nimage.threshold_img(fmriFrame, 12500)
            #view = plotting.view_img(fmriFrame, bg_img=anatTestPath)
            #view.open_in_browser()
            dataThresh, _ = mrifrac.fmriThresh(fmriFrame, thresh=t, verbose=0)
            fractalDimensionFrames[i] = mrifrac.fractalDimension3D(dataThresh, n_samples = 30, n_offsets= 10, plot=False)
            lac, _, _ = lacunarity.lacunarity3D(dataThresh, 4, 1)
            lacunarityFrames[i] = lac[0]
            # Calculate spectral power
            
            low = 1
            high = 60
            # Define window length
            nperseg = frameSize
            # Compute the modified periodogram (Welch)
            freqs, psd = welch(eegDataSplit[i, :], fs, nperseg=nperseg)
            # Frequency resolution
            freq_res = freqs[1] - freqs[0]
            # Find closest indices of band in frequency vector
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Integral approximation of the spectrum using Simpson's rule.
            spectralPowerFrames[i] = simps(psd[idx_band], dx=freq_res)
            
        # Normalize the fractal dimension to zero mean and unit variance
        fractalDimensionNormalized = fractalDimensionFrames - np.mean(fractalDimensionFrames)
        fractalDimensionNormalized = fractalDimensionNormalized / np.std(fractalDimensionNormalized);
        
        # Same for the spectral power
        spectralNormalized = spectralPowerFrames - np.mean(spectralPowerFrames)
        spectralNormalized = spectralNormalized / np.std(spectralNormalized)
        
        # Same for lacunarity
        lacunarityNormalized = lacunarityFrames - np.mean(lacunarityFrames)
        lacunarityNormalized = lacunarityNormalized / np.std(lacunarityNormalized)
        
        rPearson, pPearson = stats.pearsonr(fractalDimensionNormalized, spectralNormalized)
        rSpearman, pSpearman = stats.spearmanr(fractalDimensionNormalized, spectralNormalized)
        rKendal, pKendal = stats.kendalltau(fractalDimensionNormalized, spectralNormalized)
        #correlations.append(correlation[0, 1])
        print ("Fractal Dimension r, p for:" +
                "\nPearson: " + str(rPearson) + ", " + str(pPearson) +
                "\nSpearman: " + str(rSpearman) + ", " + str(pSpearman) +
                "\nKendal: " + str(rKendal) + ", " + str(pKendal))
        rPearson, pPearson = stats.pearsonr(lacunarityNormalized, spectralNormalized)
        rSpearman, pSpearman = stats.spearmanr(lacunarityNormalized, spectralNormalized)
        rKendal, pKendal = stats.kendalltau(lacunarityNormalized, spectralNormalized)
        #correlations.append(correlation[0, 1])
        print ("Lacunarity r, p for:" +
               "\nPearson: " + str(rPearson) + ", " + str(pPearson) +
               "\nSpearman: " + str(rSpearman) + ", " + str(pSpearman) +
               "\nKendal: " + str(rKendal) + ", " + str(pKendal))
        
        startTime = eegStartIndex / (fs * downsampleFactor)
        dataExport = pd.DataFrame({"Time(s)" : t + startTime, 
                                    "Fractal dimension" : fractalDimensionFrames, 
                                    "Fractal Dimension Standardized" : fractalDimensionNormalized, 
                                   "Lacunarity" : lacunarityFrames, 
                                   "Lacunarity Standardized" : lacunarityNormalized,
                                   "Spectral Power(uV^2 / Hz)" : spectralPowerFrames,
                                   "Spectral Power Standardized" : spectralNormalized})
        dataExport.to_excel("Analysis_" + str(t) + "t_" + str(4) + "mm.xlsx")
