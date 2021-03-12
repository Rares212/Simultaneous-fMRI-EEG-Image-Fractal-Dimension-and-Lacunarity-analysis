# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:40:17 2020

@author: Rares
"""

from nilearn.regions import Parcellations
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import numpy as np
import mloop
from mloop.data import reading
import pandas as pd

def normalizeArray(data):
    # Normalize the data to [0, 1]
    dataMax = np.max(data)
    dataMin = np.min(data)
    if (dataMin < 0):
        data = data + dataMin
        if (dataMax + dataMin > 1):
            data = data / (dataMax + dataMin)
    else:
        if (dataMax > 1):
            data = data / dataMax
    return data

def applyCmap(x, cmap):
    return cmap(x)

def volumeDataToRGBA(data, cmap, transp=0.5):
    """
    Takes a 3D volumetric array and turns it into an
    array of shape X, Y, Z, RGBA
    
    Parameters
    ----------
    data :
        3D numpy array
    cmap :
        matplotlib colormap

    Returns
    -------
    RGBA version of the data.
    """
    # Normalize the data to [0, 1]
    dataMax = np.max(data)
    dataMin = np.min(data)
    if (dataMin < 0):
        data = data + dataMin
        if (dataMax + dataMin > 1):
            data = data / (dataMax + dataMin)
    else:
        if (dataMax > 1):
            data = data / dataMax
    
    # Empty array of shape (X, Y, Z, RGBA)
    res = np.empty(data.shape + (4,))
    
    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]
    eps = 0.0000001
    # for x in range(0, nx):
    #     for y in range(0, ny):
    #         for z in range(0, nz):
    #             if (data[x, y, z] < eps):
    #                 res[x, y, z] = (0, 0, 0, 0)
    #             else:
    #                 res[x, y, z] = np.asarray(cmap(data[x, y, z])) * 255
    #                 res[x, y, z, 3] = transp * 255
    mask = data > eps
    res = np.asarray(cmap(data, transp)) * 255
    # res[:, :, :, 3][res[:, :, :, 3] > eps] = transp * 255
    res *= mask[..., np.newaxis]
    res = res.astype(np.ubyte)
    # res = res.astype(np.ubyte)
    return res

def changeVolumeTransparency(data, transp):
    # Clamp the transparency to [0, 1]
    transp = max(min(transp, 1), 0)
    # nx = data.shape[0]
    # ny = data.shape[1]
    # nz = data.shape[2]
    eps = 0.0000001
    # for x in range(0, nx):
    #     for y in range(0, ny):
    #         for z in range(0, nz):
    #             if (data[x, y, z, 3] > eps):
    #                 data[x, y, z, 3] = np.ubyte(transp * 255)
    #mask = data[:, :, :, 3] > eps
    data[:, :, :, 3][data[:, :, :, 3] > eps] = np.ubyte(transp * 255)
    return data
    

def eegSplit(eegData, excludedChannels=None, frameSize=2000):
    # (Entities, Features)
    # Entities - 2s of eeg activity
    # Features - Electrode data
    
    nx = eegData.shape[0]
    ny = eegData.shape[1]
    if (excludedChannels):
        filler = np.zeros((1, ny))
        eegData = np.insert(eegData, excludedChannels, filler, axis=0)
    nx = eegData.shape[0]
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

def eegToDataFrame(eegData, frameSize=2000, fs=1000):
    eegData = np.ndarray.transpose(eegData)
    nElectrodes = eegData.shape[1]
    nSamples = eegData.shape[0]
    nFrames = nSamples / frameSize
    
    idCol = np.zeros((nSamples, 1))
    timeCol = np.zeros((nSamples, 1))
    for i in range(0, nSamples):
        idCol[i] = int(i // frameSize)
        timeCol[i] = i / float(fs)
        # timeCol[i] = int(i - (idCol[i] * frameSize))
    colNames = [None] * (nElectrodes + 2)
    colNames[0] = "id"
    colNames[1] = "time"
    for i in range(0, nElectrodes):
        colNames[i+2] = "Electrode " + str(i)
    dfArray = np.concatenate((idCol, timeCol, eegData), axis=1)
    dfArray = pd.DataFrame(dfArray, columns=colNames)
    return dfArray
    
    
    
    
def timeAlign(fmriEventsPath, eegEventsPath, eegSampleRate=1000, fmriRepetitionTime=2):
    fmriEvents = reading.readEventsFMRI(fmriEventsPath)
    eegEvents  = reading.readEventsEEG(eegEventsPath)
    eegFirstEventIndex = eegEvents['tstim'][0][0]
    
    fmriFirstEventSeconds = float(fmriEvents[0]['onset'])
    # Rounds up to the nearest multiple of the repetition time
    fmriFirstSliceSeconds =  fmriRepetitionTime * round(fmriFirstEventSeconds / fmriRepetitionTime)
    
    fmriStartIndex = int(fmriFirstSliceSeconds / fmriRepetitionTime)
    eegStartIndex = int((fmriFirstSliceSeconds - fmriFirstEventSeconds) * eegSampleRate + eegFirstEventIndex)
    return(fmriStartIndex, eegStartIndex)

def clusterWard(img=None, nParcels=1024, standardize=False, smoothing=2):
    """
    Does brain parcellation using Ward clustering
    img -> nii image variable or path
    nParcels (optional, default 1024) -> number of parcels
    standardize (optional, default True) ->
    smoothing (optional, default 2) -> int - the higher it is, the more smoothing is applied
    Returns a tuple containing:
        1 -> Float array of shape (nScans, nParcels) - contains the parcel signals
        2 -> The ward parcellation object
    """
    ward = Parcellations(method='ward', n_parcels=nParcels,
                         standardize=standardize, smoothing_fwhm=smoothing,
                         memory='nilearn_cache', memory_level=1, verbose=1)
    ward.fit(img)
    img = ward.transform(img)
    return img, ward
    

