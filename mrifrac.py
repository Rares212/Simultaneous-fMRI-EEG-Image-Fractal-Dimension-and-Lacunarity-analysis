# -*- coding: utf-8 -*-

import os
import nibabel as nib
from nilearn import image, masking, plotting, datasets
from nilearn.input_data import NiftiMapsMasker
from sklearn.covariance import GraphicalLassoCV
from skimage import io, morphology, color, measure, filters, util, transform, segmentation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import nibabel.nifti1 as nifti

def nii2png(niiPath=None, imgPath=None, axis='z', cut=(0,), blackBg='auto', mask=1, maskThresh=0.5, maskConnected=True, maskOpening=3, verbose=0):
    """
    Takes a .nii file and converts it to a .png file containing one or multiple slices of the brain arranged horizontally
    Arguments:
    niiPath -> the .nii filepath (relative or absolute)
    imgPath -> the filepath of the output image (must end in .png)
    axis (optional, default='z') -> 'xyz' 'xy' 'xz' 'yz' 'x' 'y' 'z' - determines the axis along which the cuts are made
    cut (optional, default=(0,)) -> list of tuples of 1, 2 or 3 integers - determines the x, y, z position along which the cuts are made
    blackBg (optional, default='auto')
    mask (optional, default=1) -> int between 0 and 3 - determines the ammount of times a gray matter mask will be applied
    maskThresh (optional, default=0.8) -> float between 0 and 1 - determines the threshold used for the gray matter mask
    maskConnected (optional, default=True) -> bool - determines if the mask will keep only the largest connected components
    maskOpening (optional, default=3) -> int - number of erosions for the morphological opening
    verbose (optional, default=0) -> int - the higher it is, the more messages will be outputted to the console
    """
    # Checks if the input filepath exists and has a .nii extension
    if (os.path.isfile(niiPath) and os.path.splitext(niiPath)[1] == '.nii'):
        # Checks if the output filepath's directory is valid, if it is not a directory itself and if it has a .png extension
        if (os.path.isdir(os.path.dirname(imgPath)) and not os.path.isdir(imgPath) and os.path.splitext(imgPath)[1] == '.png'):
            imgMask = None
            brainImg = image.load_img(niiPath)
            # Clamps mask between 0 and 3
            mask = min(3, max(0, mask))
            # Applies the mask
            for i in range(0, mask):
                imgMask = masking.compute_gray_matter_mask(brainImg, threshold=maskThresh, connected=maskConnected, opening=maskOpening, verbose=verbose)
                brainImg = image.math_img('img1 * img2', img1=brainImg, img2=imgMask)
            if (verbose > 0):
                print('Outputing image at {}...'.format(imgPath))
            plotting.plot_anat(anat_img=brainImg, display_mode=axis, output_file=imgPath, cut_coords=cut, annotate=False, draw_cross=False, black_bg=blackBg)
        elif (not os.path.isdir(os.path.dirname(imgPath))):
            print('{} is an invalid directory!!!'.format(os.path.dirname(imgPath)))
        elif (os.path.isdir(imgPath)):
            print('{} is a directory, not a valid file path!!!'.format(imgPath))
        elif (not os.path.splitext(imgPath)[1] == '.png'):
            print('{} is not a .png file!!!'.format(os.path.split(imgPath)[1]))
    elif (not os.path.isfile(niiPath)):
        print('{} is not a valid path!!!'.format(niiPath))
    elif (not os.path.splitext(niiPath)[1] == '.nii'):
        print('{} is not a .nii file!!!'.format(os.path.split(niiPath)[1]))
        
def nii2csv(niiPath=None, csvPath=None, axis='z', cut=(0,), blackBg='auto', mask=1, maskThresh=0.5, maskConnected=True, maskOpening=3, verbose=0):
    """
    Takes a .nii file and converts it to a .csv file containing a 3D grayscale array 
    Arguments:
    niiPath -> the .nii filepath (relative or absolute)
    csvPath -> the filepath of the output image (must end in .png)
    axis (optional, default='z') -> 'xyz' 'xy' 'xz' 'yz' 'x' 'y' 'z' - determines the axis along which the cuts are made
    cut (optional, default=(0,)) -> list of tuples of 1, 2 or 3 integers - determines the x, y, z position along which the cuts are made
    blackBg (optional, default='auto')
    mask (optional, default=1) -> int between 0 and 3 - determines the ammount of times a gray matter mask will be applied
    maskThresh (optional, default=0.8) -> float between 0 and 1 - determines the threshold used for the gray matter mask
    maskConnected (optional, default=True) -> bool - determines if the mask will keep only the largest connected components
    maskOpening (optional, default=3) -> int - number of erosions for the morphological opening
    verbose (optional, default=0) -> int - the higher it is, the more messages will be outputted to the console
    """
    # Checks if the input filepath exists and has a .nii extension
    if (os.path.isfile(niiPath) and os.path.splitext(niiPath)[1] == '.nii'):
        # Checks if the output filepath's directory is valid, if it is not a directory itself and if it has a .png extension
        if (os.path.isdir(os.path.dirname(csvPath)) and not os.path.isdir(csvPath) and os.path.splitext(csvPath)[1] == '.png'):
            imgMask = None
            brainImg = image.load_img(niiPath)
            # Clamps mask between 0 and 3
            mask = min(3, max(0, mask))
            # Applies the mask
            for i in range(0, mask):
                imgMask = masking.compute_gray_matter_mask(brainImg, threshold=maskThresh, connected=maskConnected, opening=maskOpening, verbose=verbose)
                brainImg = image.math_img('img1 * img2', img1=brainImg, img2=imgMask)
            if (verbose > 0):
                print('Outputing image at {}...'.format(csvPath))
            plotting.plot_anat(anat_img=brainImg, display_mode=axis, output_file=csvPath, cut_coords=cut, annotate=False, draw_cross=False, black_bg=blackBg)
        elif (not os.path.isdir(os.path.dirname(csvPath))):
            print('{} is an invalid directory!!!'.format(os.path.dirname(csvPath)))
        elif (os.path.isdir(csvPath)):
            print('{} is a directory, not a valid file path!!!'.format(csvPath))
        elif (not os.path.splitext(csvPath)[1] == '.png'):
            print('{} is not a .png file!!!'.format(os.path.split(csvPath)[1]))
    elif (not os.path.isfile(niiPath)):
        print('{} is not a valid path!!!'.format(niiPath))
    elif (not os.path.splitext(niiPath)[1] == '.nii'):
        print('{} is not a .nii file!!!'.format(os.path.split(niiPath)[1]))
            
def plotComparison(original, filtered, originalName=None, filterName=None):
    """
    Plots 2 images side by side in the console
    original -> left image
    filtered -> right image
    originalName (optional) -> label for the original image
    filterName (optional) -> label for the filtered image
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title(originalName)
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filterName)
    ax2.axis('off')

def getLargestCC(segment):
    """
    Gets the largest connected component of an image
    Used to filter out the skull after thinning
    segment -> image to be processed
    """
    labels = measure.label(segment, 8, connectivity=3)
    largestCC = labels == (np.argmax(np.bincount(labels.flatten())[1:]) + 1)
    return largestCC

def fractalDimension(img, threshold=0.9):
    """
    Gets the fractal dimension of an image
    img -> input image
    threshold (optional, default=0.9) -> float between 0 and 1; determines the threshold value for binarizing the image
    """
    # Only for 2d image
    assert(len(img.shape) == 2)
    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                               np.arange(0, img.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])
    # Transform img into a binary array
    img = (img > threshold)
    # Minimal dimension of img
    p = min(img.shape)
    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))
    # Extract the exponent
    n = int(np.log(n)/np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(img, size))
    # Fit the successive log(sizes) with log(counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def fmriThresh(img, thresh="90%", mask=True, verbose=0):
    """
    Parameters
    ----------
    img : nimage
        FMRI Image or path to an image
    thresh : float [0, 1] or string, optional
        The default is "90%".
        Sets the threshold
    mask : bool, optional
        The default is True.
        Wheter to use an epi mask or not
    verbose : int, optional
        Sets wether to plot the scan or not
        

    Returns
    -------
    data : 3d binary np array
        Thresholded brain scan
    affine : np array
        For reconstructing the nimage
    """
    brain = image.load_img(img)
    if (mask):
        imgMask = masking.compute_epi_mask(epi_img=brain, connected=True, opening=1)
        brain = image.math_img('img1 * img2', img1=brain, img2=imgMask)
    #brain = image.threshold_img(brain, thresh)
    brain = image.threshold_img(brain, thresh)
    affine = brain.affine
    data = brain.get_data()
    #data = (data > filters.threshold_otsu(data)) * data
    
    if (verbose):
        #brain = nib.Nifti1Image(data.astype(int), affine)
        plotting.plot_glass_brain(brain)
    return data, affine

def fractalDimension3D(array, max_box_size = None, min_box_size = 1, n_samples = 20, n_offsets = 0, plot = False):
    """Calculates the fractal dimension of a 3D numpy array.
    
    Args:
        array (np.ndarray): The array to calculate the fractal dimension of.
        max_box_size (int): The largest box size, given as the power of 2 so that
                            2**max_box_size gives the sidelength of the largest box.                     
        min_box_size (int): The smallest box size, given as the power of 2 so that
                            2**min_box_size gives the sidelength of the smallest box.
                            Default value 1.
        n_samples (int): number of scales to measure over.
        n_offsets (int): number of offsets to search over to find the smallest set N(s) to
                       cover  all voxels>0.
        plot (bool): set to true to see the analytical plot of a calculation.
                            
        
    """
    #determine the scales to measure on
    if max_box_size == None:
        #default max size is the largest power of 2 that fits in the smallest dimension of the array:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size,min_box_size, num = n_samples, base =2 ))
    scales = np.unique(scales) #remove duplicates that could occur as a result of the floor
    
    #get the locations of all non-zero pixels
    locs = np.where(array > 0)
    voxels = np.array([(x,y,z) for x,y,z in zip(*locs)])
    
    #count the minimum amount of boxes touched
    Ns = []
    #loop over all scales
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        #search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(0, i, scale) for i in array.shape]
            bin_edges = [np.hstack([0-offset,x + offset]) for x in bin_edges]
            H1, e = np.histogramdd(voxels, bins = bin_edges)
            touched.append(np.sum(H1>0))
        Ns.append(touched)
    Ns = np.array(Ns)
    
    #From all sets N found, keep the smallest one at each scale
    Ns = Ns.min(axis=1)
   
    
    
    #Only keep scales at which Ns changed
    scales  = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])
    
    
    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[:len(Ns)]
    #perform fit
    coeffs = np.polyfit(np.log(1/scales), np.log(Ns),1)
    
    #make plot
    if plot:
        fig, ax = plt.subplots(figsize = (8,6))
        ax.scatter(np.log(1/scales), np.log(np.unique(Ns)), c = "teal", label = "Measured ratios")
        ax.set_ylabel("$\log N(\epsilon)$")
        ax.set_xlabel("$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(coeffs, np.log(1/scales))
        ax.plot(np.log(1/scales), fitted_y_vals, "k--", label = f"Fit: {np.round(coeffs[0],3)}X+{coeffs[1]}")
        ax.legend();
    return(coeffs[0])
    

def get2Dskeleton(img=None, axis = 'z', sliceIndex=0, lowThresh=0.21, highThresh=0.27, mask=True, verbose=0):
    """
    Gets the 2D skeleton of an image
    imgPath -> relative or absolute path to a .png image
    lowThresh (optional, default=0.21) -> float between 0 and 1 - low threshold for the hysteresis filter
    highThresh (optional, default=0.27) -> float between 0 and 1 - high threshold for the hysteresis filter
    crop (optional, default=(15, 143, 30, 158)) -> (x1, x2, y1, y2) Tuple containing the crop coordinates
    !!! lowThresh must be lower than highThresh
    verbose (optional, default=0) -> int - the higher it is, the more messages will be outputted to the console
    """
    img = image.load_img(img)
    if (mask):
        imgMask = masking.compute_gray_matter_mask(img, verbose=verbose)
        img = image.math_img('img1 * img2', img1=img, img2=imgMask)
    img = img.get_fdata()
    
    if (axis == 'y'):
        img = img[sliceIndex]
    elif (axis == 'x'):
        img = img[:, sliceIndex, :]
    elif (axis == 'z'):
        img = img[:, :, sliceIndex]
    
    if (verbose > 0):
        print('Applying hysterisis filter...\n')
    # Binarize the image with a hysteresis filter
    res = filters.apply_hysteresis_threshold(img, lowThresh, highThresh)
    # Get the largest connected component of the image
    if (verbose > 2):
        plotComparison(img, res, 'Original', 'L = ' + str(lowThresh) + " R = " + str(highThresh))
    # Skeletonize the image
    res = morphology.skeletonize(res)
    res = getLargestCC(res)
    res = res.astype(int)
    #print("Minkowski–Bouligand dimension: ", dimension)
    if (verbose > 1):
        print('Filtered image:')
        plotComparison(img, res, 'Original', 'Skeletonized')
    return res

def get3Dskeleton(imgPath=None, thresh='30%', mask=True, verbose=0):
    """
    Gets the 3D skeleton of an image
    imgPath -> Relative or absolute path to a .nii image or nii variable
    thresh (optional, default='30%') -> Threshold value. Either float or string like 'x.xx%'
    mask (optional, default=True) -> Applies a gray matter mask if true
    verbose (optional, default=0) -> int - the higher it is, the more messages will be outputted to the console
    Returns a Nifti1Image object containing the skeletonized image
    """
    brain = image.load_img(imgPath)
    if (mask):
        imgMask = masking.compute_gray_matter_mask(target_img=brain, threshold=0.3, connected=True, opening=1)
        brain = image.math_img('img1 * img2', img1=brain, img2=imgMask)
    brain = image.threshold_img(brain, thresh)
    affine = brain.affine
    data = brain.get_data()
    data = data / np.amax(data)
    #data = filters.threshold_adaptive(data, 35, 10)
    skeleton = morphology.skeletonize_3d(data)
    skeletonBrain = nifti.Nifti1Image(skeleton, affine) 
    return skeletonBrain

def getConnectome(imgPath=None, atlasPath=None, viewInBrowser=False, displayCovMatrix=False):
    """
    Gets the connectome of a functional MRI scan
    imgPath -> absolute or relative path to the .nii file
    atlasPath -> download path for the reference MSDL atlas
    viewInBrowser (optional, default=False) -> if True, opens up an interactive viewer in the browser
    displayCovMatrix (optional, default=False) -> display the inverse covariance matrix
    Returns a tuple of shape (estimator, atlas)
    """
    # Download the reference atlas
    atlas = datasets.fetch_atlas_msdl(data_dir=atlasPath)
    # Loading atlas image stored in 'maps'
    atlasFilename = atlas['maps']
    # Get the time series for the fMRI scan
    masker = NiftiMapsMasker(maps_img=atlasFilename, standardize=True, memory='nilearn_cache', verbose=5)
    timeSeries = masker.fit_transform(imgPath)
    # Compute the connectome using sparse inverse covariance
    estimator = GraphicalLassoCV()
    estimator.fit(timeSeries)
    if (displayCovMatrix):
        labels = atlas['labels']
        plotting.plot_matrix(estimator.covariance_, labels=labels, figure=(9, 7), vmax=1, vmin=-1, title='Covariance')
        plotting.plot_matrix(estimator.precision_, labels=labels, figure=(9, 7), vmax=1, vmin=-1, title='Inverse covariance (Precision)')
        #covPlot.get_figure().savefig('Covariance.png')
        # precPlot.get_figure().savefig('Inverse Covariance.png')
    if (viewInBrowser):
        coords = atlas.region_coords
        view = plotting.view_connectome(-estimator.precision_, coords, '60.0%')
        #view.save_as_html(file_name='Connectome Test.html')
        view.open_in_browser()
    return (estimator, atlas)

def normalize(arr):
    arrMin = np.min(arr)
    return (arr - arrMin) / (np.max(arr) - arrMin)

def explode(data):
    shapeArr = np.array(data.shape)
    size = shapeArr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shapeArr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expandCoordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plotVoxelArray(voxels, angle=320):
    voxels = normalize(voxels)

    facecolors = cm.viridis(voxels)
    facecolors[:,:,:,-1] = voxels
    facecolors = explode(facecolors)

    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expandCoordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.voxels(x, y, z, filled, facecolors=facecolors)
    plt.show()
