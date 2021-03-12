# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:54:48 2020

@author: Rares
"""

import scipy
from scipy import signal, fft
import numpy as np

def fillExcludedChannels(eegData, excludedChannels):
    nx = eegData.shape[0]
    ny = eegData.shape[1]
    filler = np.zeros((1, ny))
    if (excludedChannels):
        eegData = np.insert(eegData, excludedChannels, filler, axis=0)
    return eegData

def eegToFreqSpace(eegData, fs=1000, downsampleFactor = 4, frameSize=2000):
    # (Entities, Features)
    # Entities - 2s of eeg activity
    # Features - Electrode data
    
    nElectrodes = eegData.shape[0]
    nSamples = eegData.shape[1]
    
    eegData = scipy.signal.decimate(eegData, downsampleFactor)
    fs = fs / downsampleFactor
    frameSize = int(frameSize / downsampleFactor)
    nSamples = eegData.shape[1]
    nFrames = int(nSamples / frameSize)
    
    spectogram = np.abs(fft.rfft(eegData[:, 0:frameSize])) / frameSize
    for frame in range(1, nFrames):
        startIndex = frame * frameSize
        freqSlice = np.abs(fft.rfft(eegData[:, startIndex:startIndex + frameSize])) / frameSize
        spectogram = np.append(spectogram, freqSlice, axis=1)
    return (spectogram, fs, frameSize // 2 + 1)

def eegSplit(eegData, frameSize=2000):
    # (Entities, Features)
    # Entities - 2s of eeg activity
    # Features - Electrode data
    
    nx = eegData.shape[0]
    ny = eegData.shape[1]
    nFrames = int(ny / frameSize)
    res = np.empty((nFrames, frameSize * nx))
    for frame in range(0, nFrames):
        for electrode in range(0, nx):
            startIndex1 = electrode * frameSize
            endIndex1 = startIndex1 + frameSize
            startIndex2 = frame * frameSize
            endIndex2 = startIndex2 + frameSize
            res[frame, startIndex1:endIndex1] = eegData[electrode, startIndex2:endIndex2] 
    return res 

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import periodigram
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band


    # Compute the modified periodogram (Welch)
    freqs, psd = periodigram(data, sf)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp