#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:42:08 2024

@author: grz85
"""

import argparse as ap
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import os.path as op
import sys

from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from astropy.convolution import Box2DKernel
from astropy.io import fits
from astropy.modeling.fitting import FittingWithOutlierRemoval, LevMarLSQFitter
from astropy.modeling.models import Polynomial1D, Polynomial2D, Gaussian1D
from astropy.modeling.models import Const1D
from astropy.stats import biweight_location as biweight
from astropy.stats import mad_std, sigma_clip
from astropy.table import Table
from distutils.dir_util import mkpath
from scipy.ndimage import percentile_filter
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from skimage.registration import phase_cross_correlation




def get_script_path():
    ''' Get Tull reduction absolute path name '''
    return op.dirname(op.realpath(sys.argv[0]))

def setup_logging(logname='hpf'):
    '''Set up a logger for shuffle with a name ``hpf``.

    Use a StreamHandler to write to stdout and set the level to DEBUG if
    verbose is set from the command line
    '''
    log = logging.getLogger(logname)
    log.propagate = False
    if not len(log.handlers):
        fmt = '[%(levelname)s - %(asctime)s] %(message)s'
        fmt = logging.Formatter(fmt)

        level = logging.INFO

        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        log = logging.getLogger(logname)
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
    return log


def get_continuum(spectra, nbins=25, use_filter=False, per=5):
    '''
    Calculates the continuum for the input spectra.

    Parameters
    ----------
    spectra : 2D numpy array
        Input spectra for each fiber.
    nbins : int, optional
        Number of bins to split the spectra for continuum estimation. 
        Default is 25.
    use_filter : bool, optional
        If True, applies a percentile filter to smooth the continuum. 
        Default is False.
    per : int, optional
        Percentile for the filter if use_filter is True. Default is 5.

    Returns
    -------
    cont : 2D numpy array
        Continuum estimate for the spectra.
    '''
    
    # Split spectra into nbins and calculate the biweight for each bin
    a = np.array([biweight(f, axis=1, ignore_nan=True) 
               for f in np.array_split(spectra, nbins, axis=1)]).swapaxes(0, 1)
    
    # Calculate mean wavelength (x-axis) for each bin
    x = np.array([np.mean(xi) 
               for xi in np.array_split(np.arange(spectra.shape[1]), nbins)])
    
    # Initialize the continuum array
    cont = np.zeros(spectra.shape)
    X = np.arange(spectra.shape[1])
    
    # Loop over each fiber to compute continuum
    for i, ai in enumerate(a):
        sel = np.isfinite(ai)  # Select valid (finite) points
        if np.sum(sel) > nbins / 2.:
            # Quadratic interpolation for valid continuum points
            I = interp1d(x[sel], ai[sel], kind='quadratic', 
                         fill_value=np.nan, bounds_error=False)
            cont[i] = I(X)
        else:
            cont[i] = 0.0  # Set to zero if not enough valid points
        
        # Optionally apply a percentile filter for smoothing
        if use_filter:
            v = int(spectra.shape[1] / nbins)
            if v % 2 == 0:
                v += 1  # Ensure window size is odd for filtering
            cont[i] = percentile_filter(spectra[i], per, v)
            # Extrapolate edges of the filter
            p0 = np.polyfit(X[v:2*v], cont[i][v:2*v], 2)
            p1 = np.polyfit(X[-2*v:-v], cont[i][-2*v:-v], 2)
            cont[i][:v] = np.polyval(p0, X[:v])
            cont[i][-v:] = np.polyval(p1, X[-v:])
    return cont


def get_fiber_profile_order(image, spec, trace, order, npix=11):
    '''
    Extracts the fiber profile for a given order of a spectrum from an image.

    Parameters
    ----------
    image : 2D array
        The 2D image from which the fiber profile is extracted.
    spec : 2D array
        The 2D spectrum for the corresponding image.
    trace : 2D array
        The pixel positions (trace) of each order in the spectrum.
    order : int
        The spectral order for which to extract the fiber profile.
    npix : int, optional
        The number of pixels above or below the trace we want to collect
    Returns
    -------
    xi : list
        Horizontal pixel offsets from the trace.
    yi : list
        Normalized intensity values at each offset.
    ri : list
        The pixel position along the dispersion axis (wavelength direction).

    '''
    
    # Initialize lists to store results
    xi, yi, ri = ([], [], [])
    
    # Loop over the spectral axis (wavelength pixels)
    for j in np.arange(spec.shape[1]):
        
        # Get the center position of the trace for the given order at column j
        center = int(np.round(trace[order, j]))
        
        # Loop over pixel positions around the trace (±11 pixels)
        for i in np.arange(-npix, npix + 1):
            # Calculate the horizontal offset from the trace center
            xi.append((center + i) - trace[order, j])
            
            # Normalize the intensity of the image at this position by the spectrum
            yi.append(image[center + i, j] / spec[order, j])
            
            # Store the current position along the spectral axis
            ri.append(j)
    
    # Return the horizontal offsets, normalized intensities, and spectral positions
    return xi, yi, ri


def base_reduction(image, masterbias=None, Ncols=2048,
                   gain=0.584, readnoise=3.06):
    '''
    Base reduction for overscan subtraction, trimming, gain multiplication, 
    and error frame creation    

    Parameters
    ----------
    image : 2d numpy array
        raw fits image
    masterbias : 2d numpy array, optional
        master bias image (trimming already applied)
    Ncols : float, optional
        Number of columns. The default is 2048.
    gain : float, optional
        Detector gain. The default is 0.584.
    readnoise: float, optional
        Detector read noise. The default is 3.06.
    Returns
    -------
    None.

    '''
    # Subtract biweight of the overscan
    image = image - biweight(image[:, (Ncols+2):], ignore_nan=True)
    
    # Trim image
    image = image[:, :Ncols]  # trim the overscan 
    image = image[1:-1, 1:-1] # strip the first and last row and column
    
    # Masterbias subtraction
    if masterbias is not None:
        image[:] = image[:] - masterbias
    
    # Calculate error frame
    error = np.sqrt(readnoise**2 + np.where(image > 0., image, 0.))
    return image, error



def get_spectra(array_flt, array_trace, error_array=None, npix=11, 
                full_data=False, weighted_extraction=False):
    '''
    Extracts spectra from a flat-fielded 2D image by tracing fibers and summing 
    over pixel columns, optionally weighted extracting,
    or returning interpolated the full data.

    Parameters
    ----------
    array_flt : 2D ndarray
        Flat-fielded image from which to extract spectra.
    array_trace : 2D ndarray
        Trace positions for each fiber across the image.
    error_array : 2D ndarray, optional
        Array containing error estimates. If None, Poisson error is used. 
        Default is None.
    npix : int, optional
        Number of pixels to sum over when extracting the spectrum.
    full_data : bool, optional
        If True, returns the full interpolated data for each fiber.
    weighted_extraction : bool, optional
        If True, use profile model to extract the spectrum
    Returns
    -------
    spec : 2D ndarray
        Extracted spectra for each fiber.
    error : 2D ndarray
        Associated error values for each fiber.
    data : 3D ndarray, optional
        Interpolated full data for each fiber (returned if full_data is True).
    XV : 2D ndarray, optional
        column positions of the 3D data array
    YV : 2D ndarray, optional
        row positions corrected for the trace of the 3D data array
    '''
    
    # If no error array is provided, calculate Poisson error 
    if error_array is None:
        pois_var = np.maximum(array_flt, 0)  # Ensure no negative variance
        error_array = np.sqrt(pois_var + 3.06**2)  # Add read noise
       
    # Initialize arrays to store extracted spectra and errors
    spec = np.zeros_like(array_trace)
    error = np.zeros_like(array_trace)

    # Setup pixel indices and bounds for extraction window (npix pixels wide)
    N = array_flt.shape[0]  # Number of rows in the image
    x = np.arange(array_flt.shape[1])  # Pixel columns
    LB = int((npix + 1) / 2)  # Lower bound for extraction
    HB = -LB + npix + 1  # Upper bound for extraction
    
    # Initialize arrays to store full (multiple rows) interpolated spectra
    if full_data: 
        data = np.zeros((array_trace.shape[0], npix, array_trace.shape[1])) 
        datae = np.zeros_like(data)
        yind, xind = np.indices(array_flt.shape)  
        XV, YV = np.meshgrid(x, np.linspace(int(-npix / 2.), int(npix / 2.), 
                                            npix))  

    # Loop through each fiber trace
    for fiber in np.arange(array_trace.shape[0]):

        # Skip fibers near image edges
        if np.round(array_trace[fiber]).min() < LB:
            continue
        if np.round(array_trace[fiber]).max() >= (N - LB):
            continue

        # Get integer pixel positions of the fiber trace
        indv = np.round(array_trace[fiber]).astype(int)
        
        # Optimal extraction Horne (1986)
        if weighted_extraction:
            # Get the normalized profile across the columns
            P = measure_fiber_profile(array_flt, array_trace, fiber,
                                      npix=npix)
            # Initialize the data and error arrays for the relevant rows/cols
            V = np.zeros_like(P)
            E = np.zeros_like(P)
            for j in np.arange(-LB, HB):
                # Grab the image data from the right rows across the columns
                V[j+LB] = array_flt[indv + j, x]
                # Grab the error data from the right rows across the columns
                E[j+LB] = error_array[indv +j, x]
            
            # Use Horne (1986) to caclulate the spectrum (leaving out the mask)
            spec[fiber] = (np.sum(P * V / E**2, axis=0) / 
                           np.sum(P**2 / E**2, axis=0))
            error[fiber] = np.sqrt(np.sum(P, axis=0) / 
                                   np.sum(P**2 / E**2, axis=0))
            continue

        # Loop over the extraction window (npix pixels)
        for j in np.arange(-LB, HB):
            if j == -LB:
                # Weight for lower boundary
                w = indv + j + 1 - (array_trace[fiber] - npix / 2.)
            elif j == HB - 1:
                # Weight for upper boundary
                w = (npix / 2. + array_trace[fiber]) - (indv + j)
            else:
                # Central pixels weight = 1
                w = 1.

            # Sum pixel values for each fiber with weights
            spec[fiber] += array_flt[indv + j, x] * w
            # Accumulate error (squared) for each pixel
            error[fiber] += (error_array[indv + j, x] * w) ** 2
        
        # Final error calculation (square root of summed squared errors)
        error[fiber] = np.sqrt(error[fiber])

        # If full_data option is enabled, perform interpolation and store data
        if full_data:
            mn = int(np.min(indv) - npix / 2 - 2)
            mx = int(np.max(indv) + npix / 2 + 3)
            X = xind[mn:mx, x].ravel()  
            Y = (yind[mn:mx, x] - array_trace[fiber][np.newaxis, :]).ravel() 
            Z = array_flt[mn:mx, x].ravel()  
            Ze = error_array[mn:mx, x].ravel()  
            R = LinearNDInterpolator((X, Y), Z)
            Re = LinearNDInterpolator((X, Y), Ze)  
            log.info('Creating spectrum for fiber %i' % (fiber + 1))
            data[fiber] = R((XV, YV))  
            datae[fiber] = Re((XV, YV)) 

    # Return full data (if requested) or just spectra and errors
    if full_data:
        return spec, error, data, datae, XV, YV
    return spec, error


def get_scattered_light(image, trace, mask, percentile=3, order=6):
    '''
    Estimate the scattered light in an image using percentile filtering, 
    convolution, and interpolation, followed by polynomial fitting.

    Parameters
    ----------
    image : 2D ndarray
        The input image data where scattered light  needs to be subtracted.
    trace : 1D ndarray or array-like
        The trace of the fibers in the image.
    mask : 2D ndarray
        A binary mask indicating pixels to ignore during scattered light 
        computation (e.g., strong signals).
    percentile : float, optional
        The percentile to use in the percentile filtering step. Default is 3.
    order : int, optional
        The polynomial order for fitting the scattered light profile. Default is 6.

    Returns
    -------
    scattered_light : 2D ndarray
        The estimated scattered light component in the image.
    S : 2D ndarray
        A copy of the scattered light estimation before polynomial fitting.
    '''

    # Initialize a blank array for scattered light
    scattered_light = image * 0.
    
    # Loop over each column of the image to estimate scattered light
    for col in np.arange(image.shape[1]):
        y = image[:, col] * 1.  # Copy column data
        bottom = 0. * y         # Initialize a 'bottom' array 

        # Perform percentile filtering on the column to identify low values 
        bottom[-2:] = y[-2:]  # Copy the last two pixels unchanged
        bottom[:-2] = percentile_filter(y[:-2], percentile, size=35)
        
        # Smooth the scattered light estimation using a Gaussian kernel 
        bottom[:-2] = convolve(bottom[:-2], Gaussian1DKernel(25.), 
                               boundary='extend')

        # Store the estimated scattered light for this column
        scattered_light[:, col] = bottom

    # Mask invalid pixels (e.g., edges and signal regions) with NaN
    scattered_light[mask > 0] = np.nan
    scattered_light[0, :] = np.nan
    scattered_light[:, -1] = np.nan
    scattered_light[-1, :] = np.nan
    scattered_light[:, 0] = np.nan

    # Create a linear space to interpolate scattered light along the columns
    x = np.linspace(-1, 1, scattered_light.shape[0])

    # Loop through the columns and interpolate scattered light where needed
    for j in np.arange(1, scattered_light.shape[1] - 1):
        good = np.isfinite(scattered_light[:, j])  # Identify valid data points
        if good.sum() > 1500:  # Only interpolate if there are enough points 
            scattered_light[:, j] = np.interp(x, x[good], 
                                              scattered_light[:, j][good])

    # Save a copy of the scattered light before polynomial fitting
    S = scattered_light * 1.

    # Copy scattered light to apply polynomial fitting
    new = scattered_light * 1.
    new[mask > 0.] = np.nan  # Apply mask again
    
    # Linear space for fitting across rows
    x = np.linspace(-1, 1, new.shape[1])
    
    # Initialize a new array to hold the polynomial fit results
    scattered_light = new * 1.
    
    # Fit a polynomial to the scattered light profile row by row
    for j in np.arange(new.shape[0]):
        y = new[j] * 1.  # Copy row data
        good = np.isfinite(y)  # Identify valid data points for fitting
        if good.sum() > 1000:  # Perform fitting only if enough valid points
            p0 = np.polyfit(x[good], y[good], order)  # Fit a polynomial
            z = np.polyval(p0, x)  # Evaluate the polynomial fit over the row
            scattered_light[j] = z  # Store the polynomial fit results

    return scattered_light, S

def get_background(image, trace, picket_bias=-8, picket_height=17):
    '''
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    trace : TYPE
        DESCRIPTION.
    picket_bias : TYPE, optional
        DESCRIPTION. The default is -8.
    picket_height : TYPE, optional
        DESCRIPTION. The default is 17.

    Returns
    -------
    None.

    '''
    mask = make_mask_for_trace(image, trace, picket_bias=-8, picket_height=17)

    im = image * (1. - mask)
    x = np.arange(im.shape[0])
    back = np.zeros_like(image)
    for j in np.arange(im.shape[1]):
        y = im[:, j]
        sel = y != 0.0
        z = percentile_filter(y[sel], 50, size=50)
        model = np.interp(x, x[sel], z)
        back[:, j] = model
    return back

def find_arc_peaks(y, local_window=151, thresh_mult=3.):
    ''' Finds peaks in a given input array above threshold, thresh

    Parameters
    ----------
    y : numpy array
        1-d array with peaks
    thresh : float
        Absolute value above which a peak is returned

    Returns
    -------
    peak_loc : numpy array
        location of peaks (index units from original y array)
    peaks : numpy array
        approximate height of peak
    '''
    # Fit second order polynomial to 3 central pixels of a possible peak
    def get_peaks(flat, XN):
        YM = np.arange(flat.shape[0])
        inds = np.zeros((3, len(XN)))
        inds[0] = XN - 1.
        inds[1] = XN + 0.
        inds[2] = XN + 1.
        inds = np.array(inds, dtype=int)
        Peaks = (YM[inds[1]] - (flat[inds[2]] - flat[inds[0]]) /
                 (2. * (flat[inds[2]] - 2. * flat[inds[1]] + flat[inds[0]])))
        return Peaks

    # Find maxima with derivative
    diff_array = y[1:] - y[:-1]
    loc = np.where((diff_array[:-1] > 0.) * (diff_array[1:] < 0.))[0]
    peaks = y[loc+1]
    high_thresh = percentile_filter(y, 84, size=local_window)
    low_thresh = percentile_filter(y, 16, size=local_window)
    thresh = (high_thresh - low_thresh) / 2.
    loc = loc[peaks > thresh_mult * thresh[loc+1]] + 1

    # Get peaks using second order poly to potential maxima
    peak_loc = get_peaks(y, loc)
    peaks = y[np.round(peak_loc).astype(int)]
    return peak_loc, peaks  

def find_peaks(y, thresh_frac=.01, local_thresh=False,
               local_window=151):
    ''' Finds peaks in a given input array above threshold, thresh

    Parameters
    ----------
    y : numpy array
        1-d array with peaks
    thresh : float
        Absolute value above which a peak is returned

    Returns
    -------
    peak_loc : numpy array
        location of peaks (index units from original y array)
    peaks : numpy array
        approximate height of peak
    '''
    # Fit second order polynomial to 3 central pixels of a possible peak
    def get_peaks(flat, XN):
        YM = np.arange(flat.shape[0])
        inds = np.zeros((3, len(XN)))
        inds[0] = XN - 1.
        inds[1] = XN + 0.
        inds[2] = XN + 1.
        inds = np.array(inds, dtype=int)
        Peaks = (YM[inds[1]] - (flat[inds[2]] - flat[inds[0]]) /
                 (2. * (flat[inds[2]] - 2. * flat[inds[1]] + flat[inds[0]])))
        return Peaks

    # Find maxima with derivative
    diff_array = y[1:] - y[:-1]
    loc = np.where((diff_array[:-1] > 0.) * (diff_array[1:] < 0.))[0]
    peaks = y[loc+1]
    if local_thresh:
        thresh = percentile_filter(y, 99, size=local_window)
        loc = loc[peaks > thresh_frac * thresh[loc+1]] + 1
    else:
        loc = loc[peaks > thresh_frac * np.nanmax(y)] + 1

    # Get peaks using second order poly to potential maxima
    peak_loc = get_peaks(y, loc)
    peaks = y[np.round(peak_loc).astype(int)]
    return peak_loc, peaks   

def get_trace(image, N=25, low=10, high=1850, order=7, keyfiber=20):
    ''' 
    Computes trace for each fiber in an image by identifying fiber peaks 
    across defined column chunks.

    Parameters
    ----------
    image : 2D numpy array
        Slope image for an alpha-bright source.
    N : int, optional
        Number of column chunks used to find the average location of fibers, default is 25.
    low : int, optional
        Lower bound on peak detection within the image, default is 10.
    high : int, optional
        Upper bound on peak detection within the image, default is 1850.
    order : int, optional
        Polynomial order for fitting trace values, default is 7.
    keyfiber : int, optional
        Index for key fiber alignment within each chunk, default is 20.

    Returns
    -------
    full_trace : 2D numpy array
        Array of shape (63, 2046) containing trace points for each fiber.
    trace : 2D numpy array
        Array of detected trace points per chunk for each fiber.
    x : numpy array
        Array of average x-coordinates for each chunk.
    '''
    # Kernel setup for peak enhancement
    B = Box1DKernel(8.)  # Smoothing kernel
    G = Gaussian1DKernel(2.5)  # Gaussian kernel for refining peaks
    
    # Divide the image into N column chunks and calculate average x-coordinates
    chunks = np.array_split(image, N, axis=1)
    x = [np.mean(xi) for xi in np.array_split(np.arange(image.shape[1]), N)]
    x = np.array(x)
    
    # Starting location at the middle (last chunk)
    M = -1 + N
    
    # Find reference fiber peaks in middle chunk using biweight mean
    y = biweight(chunks[M], ignore_nan=True, axis=1)
    c = convolve(y, B)  # Enhance fiber peaks
    bottom = percentile_filter(c, 5, size=41)
    c = convolve(c - bottom, G)  # Smooth using Gaussian kernel
    ref, values = find_peaks(c, local_thresh=True)
    
    # Filter detected peaks to stay within defined bounds
    sel = (ref > low) & (ref < high)
    ref = ref[sel]
    
    # Estimate spacing between fibers and propagate adjustments
    xp = ref[:-1]
    yp = np.diff(ref)
    model = np.polyval(np.polyfit(xp, yp, 7), xp)
    for j in np.arange(keyfiber)[::-1]:
        ref[j] = ref[j + 1] - model[j]
    for j in np.arange(keyfiber + 1, len(ref)):
        ref[j] = ref[j - 1] + model[j - 1]
    
    # Build trace by aligning peaks with reference across all chunks
    trace = np.zeros((len(ref), N))
    trace[:, M] = ref

    # Loop backward through chunks, adjusting trace with reference alignment
    for i in np.arange(M)[::-1]:
        y = biweight(chunks[i], ignore_nan=True, axis=1)
        c = convolve(y, B)
        bottom = percentile_filter(c, 5, size=41)
        c = convolve(c - bottom, G)
        locs, values = find_peaks(c, local_thresh=True)
        
        # Align detected peaks to the reference locations
        D = np.abs(ref[:, np.newaxis] - locs[np.newaxis, :])
        inds = np.argmin(D, axis=1)
        ref = locs[inds] * 1.0
        xp = ref[:-1]
        yp = np.diff(ref)
        model = np.polyval(np.polyfit(xp, yp, 7), xp)
        
        # Adjust trace points for consistency across fibers
        for j in np.arange(keyfiber)[::-1]:
            ref[j] = ref[j + 1] - model[j]
        for j in np.arange(keyfiber + 1, len(ref)):
            ref[j] = ref[j - 1] + model[j - 1]
        
        trace[:, i] = ref * 1.0  # Store adjusted reference for each chunk

    # Generate full trace by fitting polynomial to trace values across chunks
    full_trace = np.zeros((trace.shape[0], image.shape[1]))
    X = np.arange(image.shape[1])
    for i in np.arange(trace.shape[0]):
        full_trace[i] = np.polyval(np.polyfit(x, trace[i, :], order), X)
    
    return full_trace, trace, x  # Return completed trace arrays


def make_mask_for_trace(image, trace, picket_height=23, picket_bias=-11):
    '''
    Creates a binary mask around the trace in the image, marking regions to be 
    ignored in further processing, based on the trace's position.

    Parameters
    ----------
    image : 2D ndarray
        The input image for which the mask is being created. 
    trace : 1D ndarray or array-like
        The vertical position of the trace in the image as a function of the 
        columns of the image. 
    picket_height : int, optional
        The vertical size (in pixels) of the mask region above and below the 
        trace. Defines how much area is masked around the trace. Default is 23.
    picket_bias : int, optional
        A vertical offset (in pixels) to shift the position of the mask 
        relative  to the trace. A positive bias moves the mask upwards, and a 
        negative bias moves it downwards. Default is -11.

    Returns
    -------
    mask : 2D ndarray
        A binary mask with the same shape as the input image. The region around 
        the trace is set to 1, indicating where the mask covers, and the rest 
        is set to 0.
    '''

    mask = 0. * image
    x = np.arange(image.shape[1])
    for y in trace:
        for i in x:
            bottom = int(y[i]) + picket_bias
            top = bottom + picket_height
            mask[bottom:top, i] = 1.
    return mask

def build_flat(image, trace, spectra, picket_height=23, picket_bias=-11):
    '''
    Constructs a flat-field correction mask for the image by normalizing 
    regions around the trace using the provided spectra data.

    Parameters
    ----------
    image : 2D ndarray
        The input image to which flat-field correction will be applied. This 
        represents the raw data that needs to be corrected.
    trace : 1D ndarray or array-like
        The vertical position of the trace in the image as a function of the 
        columns of the image. 
    spectra : 2D ndarray
        The spectral data corresponding to each trace in the image. It provides 
        the normalization factor for each pixel in the region defined by the 
        trace and picket dimensions.
    picket_height : int, optional
        The vertical size (in pixels) of the region around the trace to be 
        included in the flat-field correction. Default is 23.
    picket_bias : int, optional
        A vertical offset (in pixels) to shift the position of the region 
        relative to the trace. A positive bias moves the region upwards, 
        and a negative bias moves it downwards. Default is -11.

    Returns
    -------
    mask : 2D ndarray
        A flat-field correction mask for the image, with the same shape as the 
        input image. The region around each trace is normalized based on the 
        input spectra, while the rest of the mask is set to 1 (no correction).
    '''
    mask = 0. * image
    mask[:] = 1.
    x = np.arange(image.shape[1])
    for j, y in enumerate(trace):
        for i in x:
            bottom = int(y[i]) + picket_bias
            top = bottom + picket_height
            mask[bottom:top, i] = image[bottom:top, i] / (spectra[j, i] / 15.)
    return mask


def extend_orders(full_trace, nbelow=5, nabove=5, order=7):
    '''
    Extends the trace of an object in an image by adding extra rows both 
    below and above the original trace. This is done by fitting a polynomial 
    to the trace's derivatives and using the fit to extrapolate the trace.

    Parameters
    ----------
    full_trace : 2D ndarray
        The original trace data, where each column corresponds to a vertical 
        position of the trace in an image.
    nbelow : int, optional
        The number of rows to extend below the original trace. Default is 5.
    nabove : int, optional
        The number of rows to extend above the original trace. Default is 5.
    order : int, optional
        The order of the polynomial used to fit the trace's derivatives for 
        extrapolation. Default is 7.

    Returns
    -------
    newtrace : 2D ndarray
        The extended trace data, with the same number of columns as 
        `full_trace`, but with extra rows added above and below.

    '''

    # Initialize an array for the extended trace with added rows
    newtrace = np.zeros((full_trace.shape[0] + nbelow + nabove, 
                         full_trace.shape[1]))
    
    # Copy the original trace into the middle of the new array
    newtrace[nbelow:-nabove] = full_trace
    
    # Calculate the number of rows in the extended trace
    N = full_trace.shape[0] + nbelow

    # Loop over each column of the trace 
    for j in np.arange(full_trace.shape[1]):
        # Get the indices of the original trace
        x = np.arange(full_trace.shape[0] - 1)
        
        # Calculate the derivative (differences) between consecutive trace points
        y = np.diff(full_trace[:, j])
        
        # Fit a polynomial of the specified order to the derivative
        p0 = np.polyfit(x, y, order)
        
        # Generate an extended range of indices to fit the polynomial over
        x5 = np.arange(0 - nbelow, full_trace.shape[0] - 1 + nabove)
        
        # Evaluate the polynomial over the extended range
        model = np.polyval(p0, x5)

        # Extrapolate the trace below the original data
        for i in np.arange(nbelow)[::-1]:
            newtrace[i, j] = newtrace[i+1, j] - model[i]
        
        # Extrapolate the trace above the original data
        for i in np.arange(N, N + nabove):
            newtrace[i, j] = newtrace[i-1, j] + model[i-1]

    return newtrace


def make_mask(image, picket_height=21, picket_bias=0, mask_picket=False):
    '''
    Creates a binary mask for an image, marking specific columns and 
    a "picket fence" region for exclusion.

    Parameters
    ----------
    image : 2D ndarray
        The input image for which the mask is being created. It provides the 
        dimensions of the mask.
    picket_height : int, optional
        The vertical size (in pixels) of the masked region around the "picket 
        fence" curve. Defines how much area is masked around the fitted curve. 
        Default is 21.
    picket_bias : int, optional
        A vertical offset (in pixels) to shift the masked region around the 
        "picket fence" curve. A positive bias moves the mask upwards, and a 
        negative bias moves it downwards. Default is 0.
    mask_picket: bool, optional
        True or False value to mask the hard-coded picket feature

    Returns
    -------
    mask : 2D ndarray
        A binary mask with the same shape as the input image. Specific columns 
        and a curved region, modeled as a "picket fence", are marked with 1's 
        (to exclude them from further analysis), while the rest of the mask is 0.
    '''

    mask = 0. * image
    # Columns to mask
    badcols = [198, 199, 214, 215, 979]
    for badcol in badcols:
        mask[:, badcol] = 1.
    # Partial column(s) to mask
    badcols = [1633, 1634]
    bottom = 1109
    for badcol in badcols:
        mask[bottom:, badcol] = 1.
    if mask_picket:
        picket_fence_bottom_x = [14., 1014., 2029.]
        picket_fence_bottom_y = [1120., 1056., 1020.]
        x = np.arange(image.shape[1])
        y = np.polyval(np.polyfit(picket_fence_bottom_x, 
                                  picket_fence_bottom_y, 2), x)
        for i in x:
            bottom = int(y[i]) + picket_bias
            top = bottom + picket_height
            mask[bottom:top, i] = 1.
    return mask

def make_1d_waveoffset_plot(xind, fit, step, folder):
    '''
    Creates and saves a 1D wavelength offset plot, visualizing the relationship 
    between the x-index and the wavelength offset step, along with a fitted 
    curve.

    Parameters
    ----------
    xind : 1D array-like
        The x-axis indices (typically representing some form of order or pixel 
        index) for the data points in the plot.
    fit : function
        A function that represents the fitted curve, which takes `xind` as 
        input and returns the fitted values to be plotted.
    step : 1D array-like
        The wavelength offset values corresponding to the x-axis indices `xind`. 
        These represent the measured offsets that are plotted as points.
    folder : str
        A string representing the folder, used in the filename when saving the 
        plot. It is included in the file name to timestamp or version the plot.

    Returns
    -------
    None.
        The function does not return anything. It saves the plot as a PNG file 
        with the folder included in the filename.
    '''

    plt.figure(figsize=(8, 7))
    plt.scatter(xind, step, color='r', edgecolor='grey')
    plt.plot(xind, fit(xind), color='k')
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', zorder=3)
    ax.tick_params(axis='y', which='both', left=True, right=True)
    ax.tick_params(axis='x', which='both', bottom=True, top=True)
    ax.tick_params(axis='both', which='major', length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.minorticks_on()
    plt.ylabel(r'Wavelength Offset ($\mathrm{\AA}$)')
    plt.xlabel(r'Order')
    plt.savefig(op.join(reducfolder, 'wave_offset_1D.png'), dpi=150)

def make_2d_wavelength_offset_plot(corrections, folder):
    '''
    Creates and saves a 2D plot of wavelength offset corrections, visualizing 
    the corrections applied across the image as a heatmap.

    Parameters
    ----------
    corrections : 2D ndarray
        A 2D array of wavelength offset correction values to be visualized. 
        Each element in this array represents the correction applied to the 
        corresponding pixel in the image (or grid).
    folder : str
        A string representing the folder, used in the filename when saving the 
        plot. This ensures that the plot is labeled with the correct version or 
        timestamp.

    Returns
    -------
    None.
        The function does not return any values. It saves the 2D wavelength 
        offset plot as a PNG file, with the filename including the provided 
        `folder`.
    '''

    plt.figure(figsize=(8, 7))
    plt.imshow(corrections, aspect=36, origin='lower', 
               cmap=plt.get_cmap('coolwarm'))
    plt.colorbar()
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', zorder=3)
    ax.tick_params(axis='y', which='both', left=True, right=True)
    ax.tick_params(axis='x', which='both', bottom=True, top=True)
    ax.tick_params(axis='both', which='major', length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.minorticks_on()
    plt.xlabel(r'Column')
    plt.ylabel(r'Order')
    plt.savefig(op.join(reducfolder, 'wave_offset_2D.png'), dpi=150)
    
def get_blaze_spectra(spectra):
    '''
    

    Parameters
    ----------
    spectra : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    x = np.linspace(-1, 1, spectra.shape[1])
    B = spectra * 1.  
    order = 8  # Polynomial order for fitting the spectra

    # Step 3: Smooth the spectra using polynomial fitting
    for j in np.arange(spectra.shape[0]):  
        y = spectra[j] * 1.  
        good = np.isfinite(y)  

        # If sufficient valid points exist, perform the polynomial fitting
        if good.sum() > 50:
            # Fit an 8th order polynomial to the spectrum
            p0 = np.polyfit(x[good], y[good], order)
            z = np.polyval(p0, x)

            # Step 4: Identify and remove outliers based on the fit
            m = mad_std((y - z) / z, ignore_nan=True)  
            cut = np.max([3. * m, 0.05])  
            outlier = np.abs(y - z) > cut * z 

            # Refit the polynomial, excluding the outliers
            p0 = np.polyfit(x[good * (~outlier)], y[good * (~outlier)], order)
            z = np.polyval(p0, x)
            B[j] = z  
            B[j][np.isnan(spectra[j])] = np.nan  

    # Step 5: Rebuild the flat-field image with the smoothed spectra
    newspectra = B
    return newspectra
    
def make_flat_field(avg_ff, full_trace, mask,  picket_height=17,  
                    picket_bias=-8):
    '''
    Generates a flat-field correction image by removing scattered light, 
    extracting spectra, and fitting a polynomial to smooth the spectral data.

    Parameters
    ----------
    avg_ff : 2D ndarray
        The input flat-field image that needs correction. It contains the 
        bias-subtracted flat-field data before further adjustments.
    full_trace : 1D or 2D ndarray
        The trace positions of the spectral orders in the image. This indicates 
        where the spectral lines or orders are located in the flat-field image.
    mask : 2D ndarray
        A mask used to exclude certain regions of the image from correction. 
        Areas with a value greater than 0 in the mask are excluded from the 
        flat-field generation.

    Returns
    -------
    flat : 2D ndarray
        The final flat-field corrected image. This array has the same 
        dimensions as `avg_ff`, but with scattered light removed and the 
        spectra smoothed and corrected.
    '''

    # Step 1: Estimate and remove scattered light from the flat-field image
    back, orig = get_scattered_light(avg_ff, full_trace, mask)

    # Subtract the estimated scattered light from the original flat-field image
    image = avg_ff - back

    # Step 2: Extract the spectra from the image
    spectra, err = get_spectra(image, full_trace)


    newspectra = get_blaze_spectra(spectra)
    flat = build_flat(image, full_trace, newspectra, 
                      picket_height=picket_height,  
                      picket_bias=picket_bias)    

    # Apply the mask to exclude certain regions from the final flat-field image
    flat[mask > 0.] = np.nan

    return flat  


def get_trace_correction(spec, image, trace, nchunks=11):
    '''
    Calculates a correction for the spectral trace by fitting centroids of 
    the fiber profile across wavelength chunks.

    Parameters
    ----------
    spec : 2D ndarray
        The spectral data.
    image : 2D ndarray
        The image from which the fiber profile is extracted.
    trace : 2D ndarray
        The initial trace positions for each spectral order.
    nchunks : int, optional
        Number of wavelength chunks. Default is 11.

    Returns
    -------
    trace_cor : 2D ndarray
        Corrected trace positions.
    '''
    
    G = Gaussian1D()  # Gaussian model for profile fitting
    C = Const1D()     # Constant baseline model
    centroids = np.zeros((len(spec), nchunks))  # Centroid storage
    XK = np.zeros_like(centroids)  # Store mean wavelength per chunk

    # Loop over spectral orders to extract and fit fiber profiles
    for order in np.arange(len(spec)):
        x, y, r = get_fiber_profile_order(image, spec, trace, order)
        x, y, r = [np.array(xi) for xi in [x, y, r]]
        X = np.arange(spec.shape[1])
        xchunks = np.array_split(X, nchunks)
        
        # Fit centroids for each chunk
        for k, xchunk in enumerate(xchunks):
            sel = (r > np.min(xchunk)) & (r <= np.max(xchunk))
            XK[order, k] = np.mean(xchunk)
            inds = np.argsort(x[sel])

            xv = np.array([biweight(xi, ignore_nan=True) 
                           for xi in np.array_split(x[sel][inds], 41)])
            yv = np.array([biweight(xi, ignore_nan=True) 
                           for xi in np.array_split(y[sel][inds], 41)])
            fit = fitter(G + C, xv, yv)
            centroids[order, k] = fit.mean_0.value

    # Fit a 2D polynomial to the centroids
    P2d = Polynomial2D(4)
    xind, yind = np.indices(centroids.shape)
    or_fitted_model, _ = or_fit(P2d, xind, yind, centroids)
    centroids_model = or_fitted_model(xind, yind)

    # Calculate the correction using 1D polynomial fits
    trace_cor = trace * np.nan
    P4 = Polynomial1D(4)
    for row in np.arange(len(spec)):
        trace_cor[row] = fitter(P4, XK[row], 
                                centroids_model[row])(np.arange(spec.shape[1]))

    return trace_cor

def measure_fiber_profile(image, spec, trace, chunksize=100, npix=17):
    '''
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    spec : TYPE
        DESCRIPTION.
    trace : TYPE
        DESCRIPTION.
    chunksize : TYPE, optional
        DESCRIPTION. The default is 100.
    npix : TYPE, optional
        DESCRIPTION. The default is 11.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    yind, xind = np.indices(image.shape)
    profile = []
    xinds = np.array_split(np.arange(4, image.shape[1]-4), 
                           int(image.shape[1] / chunksize))
    mx = np.array([np.mean(xind) for xind in xinds])
    xI = np.arange(image.shape[1])
    LB = int((npix + 1) / 2)  # Lower bound for the extraction window
    N = int(LB * 4 + 1)
    T = np.zeros((len(trace), image.shape[1], N))
    cnt = 0
    for fibert, fibers in zip(trace, spec):
        profile.append([])
        xinterp = np.linspace(-LB, LB, N)
        dx = xinterp[1] - xinterp[0]
        for xind in xinds:
            indl = int(np.max([4, np.min(fibert[xind])-20.]))
            indh = int(np.min([image.shape[0]-4, np.max(fibert[xind])+20.]))
            foff = (yind[indl:indh, xind[0]:xind[-1]] -
                    fibert[np.newaxis, xind[0]:xind[-1]])
            V = (image[indl:indh, xind[0]:xind[-1]] /
                 fibers[np.newaxis, xind[0]:xind[-1]])
            sel = np.abs(foff) <= LB+2
            allx = foff[sel]
            ally = V[sel]
            inorder = np.argsort(allx)
            xbin = np.array([np.mean(chunk) for chunk in np.array_split(allx[inorder], 75)])
            ybin = np.array([biweight(chunk) for chunk in np.array_split(ally[inorder], 75)])
            back = biweight(ybin[(np.abs(xbin)>=LB) * (np.abs(xbin)<=LB+1.5)])
            ybin = ybin - back
            ybin[np.abs(xbin)>=LB] = 0.0
            I = interp1d(xbin[np.abs(xbin)<=LB+1], ybin[np.abs(xbin)<=LB+1], 
                              kind='quadratic', fill_value=0.0,
                              bounds_error=False)
            norm = I(xinterp).sum() * dx
            I = interp1d(xbin[np.abs(xbin)<=LB+1], ybin[np.abs(xbin)<=LB+1]/ norm, 
                              kind='quadratic', fill_value=0.0,
                              bounds_error=False)
            profile[-1].append(I)
        x = np.linspace(-LB, LB, N)
        sel = np.abs(x) <= 2.5
        y = np.array([p(x) for p in profile[-1]])
        ny = biweight(y[:, sel], axis=0)
        Y = np.zeros((image.shape[1], len(x)))
        Y[:, sel] = ny
        for i in np.where(~sel)[0]:
            Y[:, i] = np.interp(xI, mx, y[:, i])
        T[cnt] = Y
        cnt += 1

    return [x, T]


def make_fiber_model_image(image, trace, npix=17):
    '''
    Makes a model image of the smoothed spectra for flat fielding

    Parameters
    ----------
    image : 2D array
        The 2D image containing the raw data.
    trace : 2D array
        The positions of each fiber trace in the image.
    npix : int, optional
        The number of pixels used for the extraction window width (default is 11).

    Returns
    -------
    clean : 2D array
        The 2D image cleaned of fiber spectra
    '''
    
    # Get the spectra for the image
    spec, err = get_spectra(image, trace, npix=15, full_data=False)
    
    newspec = get_blaze_spectra(spec)
    
    model_info = measure_fiber_profile(image, newspec, trace, 
                                       npix=npix, chunksize=50)
    xv, model = model_info

    clean = np.zeros_like(image)
    M = np.zeros(image.shape, dtype=bool)
    
    # Setup pixel indices and bounds for extraction window (npix pixels wide)
    N = image.shape[0]  # Number of rows in the image (spatial axis)
    x = np.arange(trace.shape[1])  # Columns (spectral axis)
    LB = int((npix + 1) / 2)  # Lower bound for the extraction window
    HB = -LB + npix + 1  # Upper bound for the extraction window
    weights = np.zeros((HB+LB, spec.shape[1]))
    for order in np.arange(trace.shape[0]):
  
        # Skip fibers near the image edges 
        if np.round(trace[order]).min() < LB:
            continue  # Skip if trace is too close to the lower edge
        if np.round(trace[order]).max() >= (N - LB):
            continue # Skip if trace is too close to the upper edge
            
        # Get integer pixel positions of the fiber trace
        indv = np.round(trace[order]).astype(int)
        
        for j in np.arange(-LB, HB):
            foff = indv + j - trace[order]
            S = RectBivariateSpline(x, xv, model[order])
            weights[j+LB] = S.ev(x, foff)
            clean[indv + j, x] = weights[j + LB] * newspec[order]
            M[indv+j, x] = True
    clean[~M] = 0.0
    return clean


def fit_wavelength_solution(spectra):
    '''
    Fits a wavelength solution to the given spectra by matching observed arc
    lines to known reference wavelengths, adjusting for offsets, and saving 
    the result.

    Parameters
    ----------
    spectra : 2D ndarray
        The input spectra array.

    Returns
    -------
    None.
    The wavelength solution is saved as a FITS file in the designated folder.
    '''
    
    # Initialize arrays for tracking offsets and wavelength solution
    inds = np.arange(spectra.shape[0])  
    offsets = np.nan * inds  
    ioff = 0  
    W = np.nan * spectra  
    
    # Step 1: Get continuum for normalization
    cont = get_continuum(spectra, nbins=105, use_filter=True, per=35)

    # Step 2: Loop over each spectral order (excluding first and last few)
    for j in inds[2:-3]:
        i = j - 2
        offset = ioff  
        sel = ecTHAR['col1'] == j + 1  
        cols = ecTHAR['col3'][sel]  
        wave = ecTHAR['col4'][sel]  

        # Find arc peaks in the current spectrum (after removing continuum)
        xloc, yval = find_arc_peaks(spectra[i] - cont[i], thresh_mult=5.)
        
        # Match observed arc peaks to reference lines
        K = 0. * cols  
        for k, col in enumerate(cols):
            K[k] = xloc[np.argmin(np.abs(col + offset - xloc))]  

        # Estimate offset using a robust biweight location method
        offsets[i] = biweight(K - cols)

    # Step 3: Fit a polynomial to the calculated offsets
    sel = np.isfinite(offsets)
    P0 = np.polyfit(inds[sel], offsets[sel], 3)  
    offset_fit = np.polyval(P0, inds)  

    # Step 4: Calculate wavelength solution for each spectral order
    for j in inds[2:-3]:
        i = j - 2
        offset = offset_fit[i] 
        sel = ecTHAR['col1'] == j + 1  
        cols = ecTHAR['col3'][sel]
        wave = ecTHAR['col4'][sel]

        # Fit a 3rd-degree polynomial to the wavelength solution
        sol = np.polyfit((cols + offset - 1000) / 2000., wave, 3)
        wnorm = (np.arange(spectra.shape[1]) - 1000.) / 2000.  
        W[i] = np.polyval(sol, wnorm)  

    # Step 5: Refine wavelength solution across all spectral orders
    for col in np.arange(spectra.shape[1]):
        xind = np.arange(spectra.shape[0])
        sel = np.isfinite(W[:, col])
        p0 = np.polyfit(xind[sel], W[sel, col], 12)  
        W[:, col] = np.polyval(p0, xind)  

    # Step 6: Apply corrections to the wavelength solution
    corrections = spectra * np.nan  
    for row in np.arange(spectra.shape[0]):
        xind = np.arange(spectra.shape[1])
        normspectrum = np.log10(spectra[row] / cont[row])
        xloc, yval = find_arc_peaks(normspectrum, thresh_mult=2.) 

        # Select reference wavelengths within the range of this spectrum
        lw = np.min(W[row])
        hw = np.max(W[row])
        sel = (Murphy_Sw > lw) * (Murphy_Sw < hw) * (Murphy_table['col3'] > 3)
        K = np.zeros((sel.sum(),))  
        cols = np.interp(Murphy_Sw[sel], W[row], xind)  

        # Match observed peaks to reference peaks
        for i, col in enumerate(cols):
            K[i] = xloc[np.argmin(np.abs(col - xloc))]

        # Calculate predicted wavelengths from the matched peaks
        pred = np.interp(K, xind, W[row])

        # Apply corrections based on the difference between predicted and reference values
        if row == 3:
            corrections[row] = biweight(Murphy_Sw[sel] - pred, ignore_nan=True)
        else:
            or_fitted_model, mask = or_fit(P2, Murphy_Sw[sel], Murphy_Sw[sel] - pred)
            corrections[row] = or_fitted_model(W[row])

    # Step 7: Add corrections to the wavelength solution
    W += corrections

    # Step 8: Save the final wavelength solution as a FITS file
    log.info('Saving Wavelength Solution to the config Folder')
    f0 = fits.PrimaryHDU(header=f[0].header)
    f1 = fits.ImageHDU(spectra)
    f2 = fits.ImageHDU(spectra * 0.)
    f3 = fits.ImageHDU(W)
    fits.HDUList([f0, f1, f2, f3]).writeto(op.join(reducfolder,
                                                   'arc_spectra.fits'), 
                                           overwrite=True)


def get_wavelength_offset_from_archive(archive_arc, archive_wave, arc_spectra):
    '''
    Computes wavelength offsets by comparing archived arc spectra with new arc
    spectra, and applies the corrections to the wavelength solution.

    Parameters
    ----------
    archive_arc : 2D ndarray
        Archived arc spectra to be used for comparison.
    archive_wave : 2D ndarray
        Archived wavelength solution corresponding to the archived arc spectra.
    arc_spectra : 2D ndarray
        New arc spectra that need to be corrected for wavelength shifts.

    Returns
    -------
    W : 2D ndarray
        Corrected wavelength solution after applying the computed offsets.
    '''
    
    # Step 1: Compute the continuum for both archive and new spectra
    archive_cont = get_continuum(archive_arc, nbins=105, use_filter=True, 
                                  per=35)
    cont = get_continuum(arc_spectra, nbins=105, use_filter=True, per=35)

    # Step 2: Initialize step offsets array
    step = np.zeros((len(archive_arc),))

    # Step 3: Loop through each row (spectral order) and calculate wavelength shifts
    for i in np.arange(len(archive_arc)):
        z = np.log10(archive_arc[i] / archive_cont[i])  
        w = archive_wave[i]  
        dw = np.mean(np.diff(w))  
        z1 = np.log10(arc_spectra[i] / cont[i])  
        good = np.isfinite(z) * np.isfinite(z1)  
        # Use phase cross-correlation to find the shift between the two spectra
        Info = phase_cross_correlation(z[good], z1[good], normalization=None, 
                                        upsample_factor=5000)
        step[i] = Info[0][0] * dw  # Convert pixel shift to wavelength shift

    # Step 4: Fit a polynomial to the wavelength offsets
    xind = np.arange(len(step))  
    fit = fitter(P3, xind, step)  
    wave_offsets = fit(xind)  

    # Step 5: Plot the wavelength offset for visualization
    make_1d_waveoffset_plot(xind, fit, step, folder)

    # Step 6: Apply the wavelength offsets to the archived wavelength solution
    W = archive_wave + wave_offsets[:, np.newaxis]  

    # Step 7: Interpolate archived spectra to match the corrected wavelengths
    newspectrum = arc_spectra * np.nan  
    for row in np.arange(len(arc_spectra)):
        I = interp1d(archive_wave[row], archive_arc[row], kind='linear', 
                      bounds_error=False, fill_value=np.nan)  
        newspectrum[row] = I(W[row])  

    # Step 8: Compute continuum for the new interpolated spectrum
    newcont = get_continuum(newspectrum, nbins=105, use_filter=True, per=35)

    # Step 9: Calculate wavelength offsets in chunks and model the steps
    corrections = arc_spectra * np.nan  # Initialize corrections array
    chunks = 5  # Number of wavelength chunks per order
    steps = np.ones((len(arc_spectra), chunks)) * np.nan  
    wchunk = np.ones((len(arc_spectra), chunks)) * np.nan  
    P2d = Polynomial2D(4) 

    # Step 10: Loop through each spectral order to calculate the step shifts for chunks
    for row in np.arange(len(arc_spectra)):
        normspectrum = np.log10(arc_spectra[row] / cont[row]) 
        normarchive = np.log10(newspectrum[row] / newcont[row])  
        step = np.zeros((chunks,))
        i = 0

        # Loop through wavelength chunks and calculate step shifts
        for w1, chunk1, chunk2 in zip(np.array_split(W[row], chunks),
                                      np.array_split(normspectrum, chunks),
                                      np.array_split(normarchive, chunks)):
            z = chunk2  # Archive spectrum chunk
            w = w1  # Wavelength chunk
            wchunk[row, i] = np.mean(w)  # Store the mean wavelength for the chunk
            dw = np.mean(np.diff(w))  # Calculate the wavelength step size
            z1 = chunk1  # New spectrum chunk
            good = np.isfinite(z) * np.isfinite(z1)  # Only use valid data points

            # Calculate step shift using phase cross-correlation
            if good.sum() > 50:
                Info = phase_cross_correlation(z[good], z1[good], 
                                                normalization=None, 
                                                upsample_factor=5000)
                step[i] = Info[0][0] * dw  # Convert pixel shift to wavelength shift
            i += 1
        steps[row] = step 

    # Step 11: Fit a 2D polynomial to the step shifts across all orders and chunks
    xind, yind = np.indices(steps.shape)  
    or_fitted_model, mask = or_fit(P2d, xind, yind, steps) 
    steps_model = or_fitted_model(xind, yind)  

    # Step 12: Apply the modeled corrections to the wavelength solution
    for row in np.arange(len(arc_spectra)):
        corrections[row] = fitter(P1, wchunk[row], steps_model[row])(W[row])

    W = W + corrections  # Apply the corrections to the wavelength solution

    # Step 13: Plot the 2D wavelength offsets for visualization
    make_2d_wavelength_offset_plot(corrections, folder)
    
    return W  


def build_master_bias(bias_files, Nrows, Ncols, Bias_section_size):
    '''
    Constructs a master bias image by averaging multiple bias frames after 
    removing a calculated bias from a specified section.

    Parameters
    ----------
    bias_files : list of str
        List of file paths to the bias images to be processed.
    Nrows : int
        Number of rows in each bias image.
    Ncols : int
        Number of columns in each bias image (excluding the bias section).
    Bias_section_size : int
        Size of the bias section to be excluded from the averaging.

    Returns
    -------
    avg_bias : 2D ndarray
        The average bias image, excluding the edge pixels based on the 
        defined section size.
    '''
    
    # Step 1: Initialize an array to hold the bias images
    N = len(bias_files)  
    images = np.zeros((N, Nrows, Ncols + Bias_section_size))  

    # Step 2: Load bias images from the specified files
    for i, filename in enumerate(bias_files):
        f = fits.open(filename)  
        images[i] = f[0].data  

    # Step 3: Remove the calculated bias from the specified section
    images = images - biweight(images[:, :, (Ncols + 2):], 
                               axis=(1, 2), ignore_nan=True)[:, np.newaxis, 
                                                            np.newaxis]
    # Retain only the relevant columns (excluding bias section)
    images = images[:, :, :Ncols]

    # Step 4: Compute the average bias image using biweight averaging
    avg_bias = biweight(images, axis=0, ignore_nan=True)
    
    # Step 5: Trim the edges of the average bias image
    avg_bias = avg_bias[1:-1, 1:-1]  
    
    return avg_bias  


def build_master_ff(ff_files, Nrows, Ncols, Bias_section_size, avg_bias):
    '''
    Constructs a master flat-field image by averaging multiple flat-field 
    frames after bias subtraction and removal of a bias section.

    Parameters
    ----------
    ff_files : list of str
        List of file paths to the flat-field images to be processed.
    Nrows : int
        Number of rows in each flat-field image.
    Ncols : int
        Number of columns in each flat-field image (excluding the bias section).
    Bias_section_size : int
        Size of the bias section to be excluded from the flat-field data.
    avg_bias : 2D ndarray
        The master bias image to subtract from each flat-field frame.

    Returns
    -------
    avg_ff : 2D ndarray
        The averaged flat-field image after bias subtraction and edge trimming.
    '''

    # Step 1: Initialize an array to hold the flat-field images
    N = len(ff_files)  
    images = np.zeros((N, Nrows, Ncols + Bias_section_size))

    # Step 2: Load flat-field images from the specified files
    for i, filename in enumerate(ff_files):
        f = fits.open(filename)  
        images[i] = f[0].data  

    # Step 3: Subtract the bias from the specified section in each image
    images = images - biweight(images[:, :, (Ncols + 2):], 
                               axis=(1, 2), ignore_nan=True)[:, np.newaxis, 
                                                            np.newaxis]

    # Step 4: Retain only the relevant columns (excluding bias section)
    images = images[:, :, :Ncols]

    # Step 5: Trim the edges and subtract the master bias from each image
    images = images[:, 1:-1, 1:-1] - avg_bias

    # Step 6: Compute the average flat-field image using biweight averaging
    avg_ff = biweight(images, axis=0, ignore_nan=True)
    
    return avg_ff  # Return the final averaged flat-field image


def build_master_arc(arc_files, Nrows, Ncols, avg_bias):
    '''
    Constructs a master arc image by averaging multiple arc frames 
    after applying bias correction.

    Parameters
    ----------
    arc_files : list of str
        List of file paths to the arc images to be processed.
    Nrows : int
        Number of rows in each arc image.
    Ncols : int
        Number of columns in each arc image.
    avg_bias : 2D ndarray
        The master bias image to subtract from each arc frame.

    Returns
    -------
    avg_arc : 2D ndarray
        The averaged arc image after bias subtraction.
    '''

    # Step 1: Determine the number of arc files and create empty 3D array
    N = len(arc_files)  
    images = np.zeros((N, Nrows - 2, Ncols - 2))  

    # Step 2: Process each arc file to remove bias and store the resulting image
    for i, filename in enumerate(arc_files):
        f = fits.open(filename)  
        # Perform base reduction with bias subtraction
        image, error = base_reduction(f[0].data, masterbias=avg_bias)
        images[i] = image  

    # Step 3: Compute the average arc image using biweight averaging
    avg_arc = biweight(images, axis=0, ignore_nan=True)
    
    return avg_arc  # Return the final averaged arc image


def deblaze_spectra(spectra, error, blaze):
    '''
    Performs deblazing on spectral data by removing the blaze function effect.


    Parameters
    ----------
    spectra : 2D ndarray
        The raw spectral data that needs to be corrected for the blaze function.
    error : 2D ndarray
        The associated error values for the raw spectral data, which will also 
        be corrected using the blaze function.
    blaze : 2D ndarray
        The blaze function values used to correct the spectra. This represents 
        the instrumental response that varies with wavelength.

    Returns
    -------
    tuple of 2D ndarrays
        A tuple containing the deblazed spectra and the corrected error values.
    '''
    
    # Divide the spectra and error by the blaze function to correct for variations
    return spectra / blaze, error / blaze  


def get_weights_for_combine(flat_spectra, wave, newwave):
    '''
    Computes normalized weights for combining flat-field spectra.

    Parameters
    ----------
    flat_spectra : 2D ndarray
        The flat-field spectral data
    wave : 2D ndarray
        The original wavelength values corresponding to each order in 
        `flat_spectra`.
    newwave : 1D ndarray
        The new wavelength grid to which the spectra will be interpolated.

    Returns
    -------
    ratios : 2D ndarray
        The normalized weight ratios for combining the flat spectra

    '''
    
    # Calculate the continuum of the flat-field spectra
    cont = get_continuum(flat_spectra, use_filter=True, per=50)

    newspec = np.zeros((wave.shape[0], newwave.shape[0]))
    
    # Interpolate continuum values to the new wavelength grid for each order
    for order in np.arange(wave.shape[0]):
        newspec[order] = np.interp(newwave, wave[order], cont[order], 
                                   left=np.nan, right=np.nan)
    
    # Calculate normalized weight ratios for combining the spectra
    ratios = newspec / np.nansum(newspec, axis=0)[np.newaxis, :]
    
    return ratios


def combine_spectrum(spectra, error, ratios, wave, newwave):
    '''
    Combines multiple spectra into a single spectrum using weighted averaging.

    Parameters
    ----------
    spectra : 2D ndarray
        The individual spectral data.
    error : 2D ndarray
        The associated errors for each spectrum.
    ratios : 2D ndarray
        The normalized weight ratios for combining the spectra.
    wave : 2D ndarray
        The original wavelength values. 
    newwave : 1D ndarray
        The new wavelength grid to which the spectra will be interpolated.

    Returns
    -------
    combined_spectrum : 1D ndarray
        The combined spectrum after applying the weight ratios
    combined_error : 1D ndarray
        The propagated error for the combined spectrum.
    '''

    # Initialize arrays to hold the interpolated spectra and errors
    newspec = np.zeros((wave.shape[0], newwave.shape[0]))
    newerr = np.zeros((wave.shape[0], newwave.shape[0]))
    
    # Interpolate spectra and errors onto the new wavelength grid for each order
    for order in np.arange(wave.shape[0]):
        sel = np.isfinite(spectra[order])
        newspec[order] = PchipInterpolator(wave[order][sel], spectra[order][sel],
                                           extrapolate=True)(newwave)
        newerr[order] = PchipInterpolator(wave[order][sel], error[order][sel],
                                           extrapolate=True)(newwave)

    
    # Combine the interpolated spectra using the provided weight ratios
    combined_spectrum = np.nansum(ratios * newspec, axis=0)
    # Calculate the combined error using the propagation of uncertainties
    combined_error = np.sqrt(np.nansum(ratios * newerr**2, axis=0))

    return combined_spectrum, combined_error


def write_spectrum(original, image, image_e, image_m, spectra, error, wave, mask, 
                   combined_wave, combined_error, combined_spectrum, header,
                   filename, data, datae):
    '''
    Saves the processed spectrum and related data into a FITS file with 
    appropriate headers and extensions.

    Parameters
    ----------
    original : 2D ndarray
        Original image.
    image : 2D ndarray
        The bias-subtracted and scattered light-corrected image.
    image_e : 2D ndarray
        The error map corresponding to `image`.
    image_m : 2D ndarray
        The mask indicating regions affected by cosmic rays or other artifacts.
    spectra : 2D ndarray
        The extracted spectra from the image for each order.
    error : 2D ndarray
        The error associated with the extracted spectra.
    wave : 2D ndarray
        The wavelength solution for each spectral order.
    mask : 2D ndarray
        The mask used during spectral extraction.
    combined_wave : 1D ndarray
        The combined wavelength array after merging different orders.
    combined_error : 1D ndarray
        The propagated error for the combined spectrum.
    combined_spectrum : 1D ndarray
        The final combined spectrum after merging all spectral orders.
    header : FITS header object
        The header containing metadata for the FITS file.
    filename : str
        The output filename where the reduced FITS file will be saved.

    Returns
    -------
    None.

    Notes
    -----
    This function updates the header with processing history and writes all 
    relevant data to a new FITS file, including the image, spectra, errors, 
    and the combined spectrum.
    '''
    
    # Add history entries to the FITS header for tracking processing steps
    header['history'] = 'Bias subtracted'
    header['history'] = 'Scatter Light Subtracted'
    header['history'] = 'Trace Adjusted'
    header['history'] = 'Flat-field Corrected'
    if not reframe_from_cosmic_rejection:
        header['history'] = 'Cosmic Ray Rejected'
    if mask_fill:
        header['history'] = 'Masked Pixels Filled'
    header['history'] = 'Fixed Aperture Extracted'
    header['history'] = 'Spectra Deblazed'
    header['history'] = 'Spectra Flattened'
    if not args.dont_cont_normalize:
        header['history'] = 'Flattened Spectrum Normalized'
    if args.full_aperture_extraction:
        header['history'] = 'Full Aperture Extracted and Appended'

    # Create a list of FITS HDUs (Header Data Units) for each data component
    L = [fits.PrimaryHDU(header=header),   # Primary HDU with header
         fits.ImageHDU(original),          # Original image
         fits.ImageHDU(image),             # Bias-subtracted image
         fits.ImageHDU(image_e),           # Error image
         fits.ImageHDU(np.array(image_m > 0., dtype=float)),  # Mask as binary
         fits.ImageHDU(spectra),           # Extracted spectra
         fits.ImageHDU(error),             # Spectra errors
         fits.ImageHDU(wave),              # Wavelength solution for each order
         fits.ImageHDU([combined_wave, combined_spectrum, 
                        combined_error])]  # Combined data
    
    if data is not None:
        L.append(fits.ImageHDU(data))
    if datae is not None:
        L.append(fits.ImageHDU(datae))
    
    # Build the FITS file name by appending '_reduced.fits' to the filename
    name = op.join(reducfolder, op.basename(filename)[:-5] + '_reduced.fits')
    
    # Write the FITS file to disk, overwriting if the file already exists
    hdulist = fits.HDUList(L)
    hdulist.writeto(name, overwrite=True)

    
def plot_trace_offset(trace_cor, name):
    '''
    Plots the residuals of the trace correction and saves the figure.

    Parameters
    ----------
    trace_cor : 2D ndarray
        The array of trace correction residuals to be plotted.
    name : str
        A name or identifier used for saving the plot.

    Returns
    -------
    None.

    '''
    
    # Create a figure with specified size
    plt.figure(figsize=(8, 7))

    # Display the trace correction residuals as a coolwarm heatmap 
    plt.imshow(trace_cor, aspect=36, origin='lower', 
               cmap=plt.get_cmap('coolwarm'))
    plt.colorbar()

    # Adjust the appearance of the ticks on both axes
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', zorder=3)
    ax.tick_params(axis='y', which='both', left=True, right=True)
    ax.tick_params(axis='x', which='both', bottom=True, top=True)
    ax.tick_params(axis='both', which='major', length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.minorticks_on()

    # Label the axes
    plt.xlabel('Column')
    plt.ylabel('Order')

    # Save the plot as a PNG file with the given name
    plt.savefig(op.join(reducfolder, 'trace_offset_%s_residual.png' % name))
    

    
# =============================================================================
# MAIN PROGRAM
# =============================================================================
# =============================================================================
# 1. Collect files
# =============================================================================


parser = ap.ArgumentParser(add_help=True)

parser.add_argument("folder", help='''folder for reduction''',
                    type=str)

parser.add_argument("rootdir",
                    help='''base directory for raw data''',
                    type=str)

parser.add_argument("-ea", "--extraction_aperture",
                    help='''Pixel aperture in "rows" used for extraction (default=11)''',
                   type=int, default=11)

parser.add_argument("-we", "--weighted_extraction",
                    help='''Extract spectra using fiber profile weights''',
                    action="count", default=0)

parser.add_argument("-fae", "--full_aperture_extraction",
                    help='''Extract the full aperture for each spectrum''',
                    action="count", default=0)

parser.add_argument("-fw", "--fit_wave",
                    help='''Fit the wavelength solution''',
                    action="count", default=0)

parser.add_argument("-drc", "--dont_reject_cosmics",
                    help='''Do not reject cosmic rays''',
                    action="count", default=0)

parser.add_argument("-dcn", "--dont_cont_normalize",
                    help='''Do not continuum normalize''',
                    action="count", default=0)

parser.add_argument("-bl", "--bias_label",
                    help='''The objet name for bias files''',
                   type=str, default='Bias')

parser.add_argument("-al", "--arc_label",
                    help='''The objet name for arc files''',
                   type=str, default='Th-Ar + CBF')

parser.add_argument("-fl", "--flat_label",
                    help='''The objet name for flat files''',
                   type=str, default='FF + CBF')

args = None
args = parser.parse_args(args=args)

# Turn off annoying warnings (even though some deserve attention)
warnings.filterwarnings("ignore")

fit_wave = args.fit_wave

# Murphy et al. (2007)
Murphy_table = Table.read('thar_MM201006.dat', format='ascii')
Murphy_Sw = np.array(Murphy_table['col2'])

ecTHAR = Table.read('ecTHAR.txt', format='ascii')

sns.set_style('ticks')
sns.set_context('talk')

plt.rcParams["font.family"] = "Times New Roman"


log = setup_logging('tull')

reframe_from_cosmic_rejection = args.dont_reject_cosmics

try:
    from maskfill import maskfill
    mask_fill = True
except:
    log.warning('maskfill is not installed')
    mask_fill = False

CUR_DIR = get_script_path()

folder = args.folder # '20240511'
basefolder = args.rootdir # '/Users/grz85/work/TS23/2.7m'
reducfolder = op.join(basefolder, folder, 'reduce')
configfolder = op.join(CUR_DIR, 'config')
mkpath(op.join(reducfolder))

path = op.join(basefolder, folder, '*.fits')
filenames = sorted(glob.glob(path))
names = []
calstage = []
for filename in filenames:
    f = fits.open(filename)
    names.append(f[0].header['OBJECT'])
    calstage.append(f[0].header['calstage'])


labels = ['bias_label', 'arc_label', 'flat_label']
for label in labels:
    setattr(args, label, getattr(args,label).lower())

# =============================================================================
# 2. Configuration values
# =============================================================================
Ncols = 2048
Nrows = 2048
Bias_section_size = 32
gain=0.584
readnoise=3.06

# =============================================================================
# 3. Master Bias Creation
# =============================================================================
log.info('Building Master Bias')
bias_files = [filename for filename, name in zip(filenames, names) 
              if name.lower() == args.bias_label]

avg_bias = build_master_bias(bias_files, Nrows, Ncols, Bias_section_size)

fits.PrimaryHDU(avg_bias, header=f[0].header).writeto(op.join(reducfolder,
                                                     'bias_image.fits'), 
                                                      overwrite=True)

# =============================================================================
# 4. Master Flat Creation
# =============================================================================
log.info('Building Master Flat Frame')

ff_files = [filename for filename, name in zip(filenames, names) 
              if name.lower() == args.flat_label]

avg_ff = build_master_ff(ff_files, Nrows, Ncols, Bias_section_size, avg_bias)
fits.PrimaryHDU(avg_ff, header=f[0].header).writeto(op.join(reducfolder,
                                                            'ff_image.fits'), 
                                                    overwrite=True)

# =============================================================================
# 5. Make Mask Frame
# =============================================================================
log.info('Building Mask Frame')
mask = make_mask(avg_ff)
fits.PrimaryHDU(mask, header=f[0].header).writeto(op.join(reducfolder,
                                                          'mask_image.fits'), 
                                                  overwrite=True)

# =============================================================================
# 6. Get Trace
# =============================================================================
log.info('Measuring the Trace from the Flat Frame')
full_trace, trace, x = get_trace(avg_ff)
fits.PrimaryHDU(full_trace, header=f[0].header).writeto(op.join(reducfolder,
                                                        'trace_image.fits'), 
                                                  overwrite=True)

# =============================================================================
# 7. Flat Field Correction
# =============================================================================
log.info('Creating the 2D Flat Field')
back, orig = get_scattered_light(avg_ff, full_trace, mask)
model = make_fiber_model_image(avg_ff - back, full_trace, npix=21)
M = make_mask_for_trace(avg_ff, full_trace, picket_height=17, picket_bias=-8)
outside_trace_sel = M == 0.
flat = (avg_ff - back) / model
flat = flat / biweight(flat[~outside_trace_sel], ignore_nan=True)
flat[outside_trace_sel] = 1.
flat[mask > 0.] = np.nan
fits.PrimaryHDU(flat, header=f[0].header).writeto(op.join(reducfolder,
                                                          'ff_model.fits'), 
                                                    overwrite=True)

# =============================================================================
# 8. Make Master Arc Image and Spectra
# =============================================================================
log.info('Building the Master Arc Frame and Spectra')
arc_files = [filename for filename, name in zip(filenames, names) 
              if name.lower() == args.arc_label]

avg_arc = build_master_arc(arc_files, Nrows, Ncols, avg_bias)
back, orig = get_scattered_light(avg_arc, full_trace, mask)

fits.PrimaryHDU(avg_arc, header=f[0].header).writeto(op.join(reducfolder,
                                                             'arc_image.fits'), 
                                                    overwrite=True)

arc_spectra, err = get_spectra(avg_arc, full_trace)
arc_cont = get_continuum(arc_spectra, nbins=105, use_filter=True, per=35)


# =============================================================================
# 9. Fitting the Arc Spectra for Wavelength Solution
# =============================================================================

fitter = LevMarLSQFitter()
or_fit = FittingWithOutlierRemoval(fitter, sigma_clip,
                                   niter=3, sigma=3.0)
P2 = Polynomial1D(2)
P1 = Polynomial1D(1)
P3 = Polynomial1D(3)


if fit_wave:
    log.info('Fitting the Wavelength Solution')
    fit_wavelength_solution(arc_spectra)
    

# =============================================================================
# 10. Load the Wavelength Solution
# =============================================================================
log.info('Adjusting the Wavelength Solution from the Archive')
L = fits.open(op.join(configfolder, 'arc_spectra.fits'))
archive_arc = L[1].data
archive_wave = L[3].data


# Adding the offset to the archive wavelength
wave = get_wavelength_offset_from_archive(archive_arc, archive_wave,
                                          arc_spectra)

# =============================================================================
# 11. Building combined wavelength for rectification
# =============================================================================
log.info('Getting combined wavelength')
wmin = np.min(np.log(wave))
wmax = np.max(np.log(wave))
wstep = np.median(np.diff(np.log(wave), axis=1))
nsteps = int((wmax - wmin) / wstep) + 1
combined_wave = np.exp(np.linspace(wmin, wmax, nsteps))

# =============================================================================
# 12. Build deblazing function and ratios for combining orders
# =============================================================================
log.info('Building the deblazing function')
back, orig = get_scattered_light(avg_ff, full_trace, mask)
ff_spectra, ff_err = get_spectra((avg_ff - back) / flat, full_trace, npix=15)
ff_cont = get_continuum(ff_spectra, use_filter=True, per=50)
blaze = ff_cont / np.nanmedian(ff_cont)
ratios = get_weights_for_combine(ff_spectra, wave, combined_wave)


# =============================================================================
# 13. Reducing science frames
# =============================================================================
log.info('Reducing the Science Frames')  

# Select filenames where calibration stage is 'OUT'
sci_files = [filename for filename, calst in zip(filenames, calstage) 
             if calst == 'OUT']

# Process the science frames
for filename in sci_files:
    f = fits.open(filename)  
    basename = op.basename(filename)[:-5]  
    
    # Perform basic image reduction using the master bias frame
    image, error = base_reduction(f[0].data, masterbias=avg_bias)
    original = image * 1.
    
    # Get scattered light correction
    back, orig = get_scattered_light(image, full_trace, mask)
    image = image - back
    
    # Apply flat-field correction
    image, error = (image / flat, error / flat)
    
    # Extract spectra from the image, using a pixel trace with npix=11
    spec, err = get_spectra(image, full_trace, npix=11, full_data=False)
    
    # Correct the trace based on the extracted spectrum
    trace_cor = get_trace_correction(spec, image, full_trace)
    
    # Plot the trace offset for diagnostic purposes
    plot_trace_offset(trace_cor, basename)
    
    trace = full_trace + trace_cor  # Update the trace

    # Subtracted the background once again
    back = get_background(image, trace, picket_bias=-8, picket_height=17)
    image = image - back
    
    # Cosmic Ray rejection using the natural smoothness of the error image
    if not reframe_from_cosmic_rejection:  # If cosmic ray detection is enabled
        smooth_error = get_continuum(error, nbins=200, use_filter=True, per=50)
        model = (error-smooth_error) / error
        model[:, :10] = 0.0
        model[:, -10:] = 0.0
        model = np.array((model > 0.5))
        model = convolve(model, Box2DKernel(2))
        total_mask = mask + (model > 0.05) # Update mask to include cosmic ray pixels
    else:
        total_mask = mask

    if mask_fill:  # If mask filling is enabled
        log.info('Filling masked pixels in image for %s' % basename)
        (image, dummy) = maskfill(image, total_mask)  # Fill masked pixels
    
    
    # Re-extract spectra using the corrected trace
    result = get_spectra(image, trace, npix=args.extraction_aperture, 
                            full_data=args.full_aperture_extraction,
                            weighted_extraction=args.weighted_extraction)
 
    if args.full_aperture_extraction:
        spec, err, data, datae, XV, YV = result
    else:
        spec, err = result
        data = None
        datae = None
    
    # Deblaze the spectra using a blaze correction
    deblazed, deblazed_err = deblaze_spectra(spec, err, blaze)
    
    # Combine the deblazed spectra and errors across different orders
    combined_spectrum, combined_err = combine_spectrum(deblazed, deblazed_err, 
                                                   ratios, wave, combined_wave)
    
    # Flatten Spectrum
    if not args.dont_cont_normalize:
        cont = get_continuum(combined_spectrum[np.newaxis, :], nbins=201, 
                             use_filter=True, per=80)
        combined_spectrum = combined_spectrum / cont[0]
        combined_err = combined_err / cont[0]
    
    # Write the final spectrum and associated data to a file
    write_spectrum(original, image, error, total_mask, spec, err, wave, mask, 
                   combined_wave, combined_err, combined_spectrum, 
                   f[0].header, filename, data, datae)
