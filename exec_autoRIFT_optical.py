#!/usr/bin/env python3

# Oversample ratio should be 128 or user-specd
# Intelligent oversample ratio setting
# Check all nodata values are included

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Yang Lei, Rob Zinke
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import re
import warnings
from osgeo import gdal
from datetime import datetime, timedelta

import numpy as np
import time
import os

from autoRIFT import __version__ as version

from autoRIFT import autoRIFT
import numpy as np
import time
import subprocess


def cmdLineParse():
    '''
    Command line parser.
    '''
    import argparse

    SUPPORTED_MISSIONS = ['S1', 'S2', 'L4', 'L5', 'L7', 'L8', 'L9']

    parser = argparse.ArgumentParser(description='Output geo grid')
    parser.add_argument(
        '-m', '--input_m', dest='indir_m', type=str, required=True,
        help='Input master image file name (in ISCE format and radar coordinates) or Input master image file name (in GeoTIFF format and Cartesian coordinates)')
    parser.add_argument(
        '-s', '--input_s', dest='indir_s', type=str, required=True,
        help='Input slave image file name (in ISCE format and radar coordinates) or Input slave image file name (in GeoTIFF format and Cartesian coordinates)')
    parser.add_argument(
        '-g', '--input_g', dest='grid_location', type=str, required=False,
        help='Input pixel indices file name')

    parser.add_argument(
        '-filtwidth', '--filter_width', dest='filter_width', type=int,
        default=5,
        help='Preprocessing filter width for Wallis filter. Default 5.')
    parser.add_argument(
        '-sr', '--search_range', dest='search_range', type=int,
        default=25,
        help='Search range limit (integer nb pixels). Default 25.')
    parser.add_argument(
        '-csmin', '--input_csmin', dest='chip_size_min', type=int,
        default=16,
        help='Input chip size min (integer power of 2). Default 16.')
    parser.add_argument(
        '-csmax', '--input_csmax', dest='chip_size_max', type=int,
        default=64,
        help='Input chip size max (integer power of 2). Default 64.')
    parser.add_argument(
        '-or', '--oversample_ratio', dest='oversample_ratio', type=int,
        default=64,
        help='Oversample ratio for subpixel measurement (integer power of 2). Default 64.')
    parser.add_argument(
        '-sf', '--input_scale_factor', dest='scale_factor', type=float,
        default=1,
        help='Input map projection scale factor file name. Default 1.0.')

    parser.add_argument(
        '-mpflag', '--mpflag', dest='mpflag', type=int, required=False,
        default=0,
        help='Number of threads for multiple threading (default is specified by 0, which uses the original single-core version and surpasses the multithreading routine)')

    return parser.parse_args()


def loadProductOptical(file_m, file_s):
    """Load the product using Product Manager.
    """
    from geogrid import GeogridOptical

    obj = GeogridOptical()

    (x1a,
     y1a,
     xsize1,
     ysize1,
     x2a,
     y2a,
     xsize2,
     ysize2,
     trans) = obj.coregister(file_m, file_s)

    DS1 = gdal.Open(file_m)
    DS2 = gdal.Open(file_s)

    I1 = DS1.ReadAsArray(xoff=x1a, yoff=y1a, xsize=xsize1, ysize=ysize1)
    I2 = DS2.ReadAsArray(xoff=x2a, yoff=y2a, xsize=xsize2, ysize=ysize2)

    I1 = I1.astype(np.float32)
    I2 = I2.astype(np.float32)

    DS1 = None
    DS2 = None

    return I1, I2


def runAutorift(
    I1, I2, xGrid, yGrid, Dx0, Dy0, SR, CSMIN, CSMAX, noDataMask,
    nodata, mpflag, geogrid_run_info, preprocessing_methods,
    preprocessing_filter_width, oversample_ratio):
    '''
    Wire and run geogrid:

    Set attributes for
        image arrays
        geographic grids
        nodata mask
        search range limit
        chip size
        oversample ratio
        initial offsets - currently all zero

    Preprocessing

    Return select attributes of the GeogridOptical object, and the nodataMask.

    Arguments:  I1, I2 - 2D arrays of image values
                xGrid, yGrid - 2D arrays of position values
                Dx0, Dy0 - 2D arrays of initial offsets (currently None)
                SR - int, search range
                CSMIN, CSMAX - int, min/max chip sizes
                noDataMask - 2D Boolean array
                nodata - float, no data value
                mpflag - int, number of threads for multiple threading
                geogrid_run_info - None
                preprocessing_methods - str, preprocessing method/filter type
                preprocessing_filter_width - int, filter width
                oversample_ratio - int, oversample ratio for subpixel proc
    Returns:    None
    '''
    # Instantiate autoRIFT object
    obj = autoRIFT()

    # Set filter width
    obj.WallisFilterWidth = preprocessing_filter_width

    # Set multiprocessing flag
    obj.MultiThread = mpflag

    # Set image arrays
    obj.I1 = I1
    obj.I2 = I2

    # Set grid arrays
    if xGrid is None:
        print('Setting grid')
        # Create the grid if it does not exist
        m, n = obj.I1.shape
        xGrid = np.arange(
            obj.SkipSampleX+10, n - obj.SkipSampleX, obj.SkipSampleX)
        yGrid = np.arange(
            obj.SkipSampleY+10, m - obj.SkipSampleY, obj.SkipSampleY)
        nd = xGrid.__len__()
        md = yGrid.__len__()
        obj.xGrid = np.int32(
            np.dot(np.ones((md, 1)), np.reshape(xGrid, (1, xGrid.__len__()))))
        obj.yGrid = np.int32(
            np.dot(np.reshape(yGrid, (yGrid.__len__(),1)), np.ones((1, nd))))
        noDataMask = np.logical_not(obj.xGrid)
    else:
        print('Using pre-determined grid')
        obj.xGrid = xGrid
        obj.yGrid = yGrid

    # Generate the nodata mask where offset searching will be skipped based on
    #        1) imported nodata mask and/or 2) zero values in the image
    # NOTE: This assumes the zero values in the image are only outside the
    #        valid image "frame", but is not true for Landsat-7 after the
    #        failure of the Scan Line Corrector, May 31, 2003.
    #        We should not mask based on zero values in the L7 images as this
    #        percolates into SearchLimit{X,Y} and prevents autoRIFT from
    #        looking at large parts of the images, but untangling the logic
    #        here has proved too difficult, so lets just turn it off if
    #        `wallis_fill` preprocessing is going to be used.
    if 'wallis_fill' not in preprocessing_methods:
        for ii in range(obj.xGrid.shape[0]):
            for jj in range(obj.xGrid.shape[1]):
                if (obj.yGrid[ii,jj] != nodata) & (obj.xGrid[ii,jj] != nodata):
                    # Mask zeros values in the image
                    if (I1[obj.yGrid[ii,jj]-1,obj.xGrid[ii,jj]-1]==0) \
                            | (I2[obj.yGrid[ii,jj]-1,obj.xGrid[ii,jj]-1]==0):
                        noDataMask[ii,jj] = True

    # Set search limits
    obj.SearchLimitX = obj.SearchLimitY = SR * np.logical_not(noDataMask)

    # Set chip size
    obj.ChipSizeMaxX = CSMAX
    obj.ChipSizeMinX = CSMIN
    obj.ChipSize0X = CSMIN

    # Set oversample ratio
    obj.OverSampleRatio = oversample_ratio

    # Set initial offsets
    obj.Dx0 = obj.Dx0 * np.logical_not(noDataMask)
    obj.Dy0 = obj.Dy0 * np.logical_not(noDataMask)

    # Replace the nodata values with zero
    obj.xGrid[noDataMask] = 0
    obj.yGrid[noDataMask] = 0


    ## Preprocessing
    t1 = time.time()
    print("Pre-process Start!!!")

    if 'wallis_fill' in preprocessing_methods:
        print(f"Using Wallis Filter Width: {obj.WallisFilterWidth}")
        obj.preprocess_filt_wal_nodata_fill()
    elif 'wallis' in preprocessing_methods:
        print(f"Using Wallis Filter Width: {obj.WallisFilterWidth}")
        obj.preprocess_filt_wal()
    elif 'fft' in preprocessing_methods:
        # FIXME: The Landsat 4/5 FFT preprocessor looks for the image corners to
        #        determine the scene rotation, but Geogrid + autoRIFT rond the
        #        corners when co-registering and chop the non-overlapping corners
        #        when subsetting to the common image overlap. FFT filer needs to
        #        be applied to the native images before they are processed by
        #        Geogrid or autoRIFT.
        warnings.warn('FFT filtering must be done before processing with'
                      'geogrid! Be careful when using this method',
                      UserWarning)
    else:
        obj.preprocess_filt_hps()
        print('Applying HPS filter')
    print("Pre-process Done!!!")
    print(f"finished in {time.time()-t1:.3f} s")

    t1 = time.time()
    obj.uniform_data_type()
    print("Uniform Data Type Done!!!")
    print(f"finished in {time.time()-t1:.3f} s")


    ## Run autoRIFT
    t1 = time.time()
    print("AutoRIFT Start!!!")
    obj.runAutorift()
    print("AutoRIFT Done!!!")
    print(f"finished in {time.time()-t1:.3f} s")

    import cv2
    kernel = np.ones((3,3),np.uint8)
    noDataMask = cv2.dilate(noDataMask.astype(np.uint8),kernel,iterations = 1)
    noDataMask = noDataMask.astype(np.bool_)

    return (obj.Dx,
            obj.Dy,
            obj.InterpMask,
            obj.ChipSizeX,
            obj.GridSpacingX,
            obj.ScaleChipSizeY,
            obj.SearchLimitX,
            obj.SearchLimitY,
            obj.origSize,
            noDataMask)


def generateAutoriftProduct(
    indir_m:str, indir_s:str, grid_location:str, filter_width:int,
    chip_size_min:int, chip_size_max:int, search_range:int,
    oversample_ratio:int, scale_factor:float, mpflag:int):
    """Heavy lifting begins here.
    Load optical products.
    Set parameters based on products and user-spec'd inputs
    Pre-process images, e.g., using Wallis filter

    Arguments:  All input arguments
    Returns:    None
    """

    # Load optical products
    data_m, data_s = loadProductOptical(indir_m, indir_s)


    ## Set parameters
    # Auxiliary parameters
    geogrid_run_info = None
    Dx0 = None  # initial offset
    Dy0 = None
    SSM = None  # stable surface mask

    # Parse grid location
    ds = gdal.Open(grid_location)
    tran = ds.GetGeoTransform()
    proj = ds.GetProjection()
    srs = ds.GetSpatialRef()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    xGrid = band.ReadAsArray()
    noDataMask = (xGrid == nodata)
    band = ds.GetRasterBand(2)
    yGrid = band.ReadAsArray()
    band = None
    ds = None

    # Parse search range
    SR = search_range
    SR = search_range

    # Parse chip size
    CSMIN = chip_size_min
    CSMAX = chip_size_max


    ## Preprocessing
    # File names
    m_name = os.path.basename(indir_m)
    s_name = os.path.basename(indir_s)

    # Set processing filter width
    preprocessing_filter_width = filter_width
    print(f"Preprocessing filter width {preprocessing_filter_width:d}")

    # Set preprocessing methods
    preprocessing_methods = ['hps', 'hps']  # for LS08 and S2
    for ii, name in enumerate((m_name, s_name)):
        # For LS07
        if len(re.findall("L[EO]07_", name)) > 0:
            acquisition = datetime.strptime(name.split('_')[3], '%Y%m%d')
            if acquisition >= datetime(2003, 5, 31):
                preprocessing_methods[ii] = 'wallis_fill'

        # For LS04-5
        elif len(re.findall("LT0[45]_", name)) > 0:
            preprocessing_methods[ii] = 'fft'
    print(f"Using preprocessing methods {preprocessing_methods}")


    ## Run autoRIFT
    # Set autoRIFT parameters
    run_params = {'I1': data_m,
                  'I2': data_s,
                  'xGrid': xGrid,
                  'yGrid': yGrid,
                  'Dx0': Dx0,
                  'Dy0': Dy0,
                  'SR': SR,
                  'CSMIN': CSMIN,
                  'CSMAX': CSMAX,
                  'noDataMask': noDataMask,
                  'nodata': nodata,
                  'mpflag': mpflag,
                  'geogrid_run_info': geogrid_run_info,
                  'preprocessing_methods': preprocessing_methods,
                  'preprocessing_filter_width': preprocessing_filter_width,
                  'oversample_ratio': oversample_ratio}

    # Run autoRIFT
    (Dx,
     Dy,
     InterpMask,
     ChipSizeX,
     GridSpacingX,
     ScaleChipSizeY,
     SearchLimitX,
     SearchLimitY,
     origSize,
     noDataMask) = runAutorift(**run_params)


    ## Save to file
    # Format output parameters
    DX = np.zeros(origSize,dtype=np.float32) * np.nan
    DY = np.zeros(origSize,dtype=np.float32) * np.nan
    INTERPMASK = np.zeros(origSize,dtype=np.float32)
    CHIPSIZEX = np.zeros(origSize,dtype=np.float32)
    SEARCHLIMITX = np.zeros(origSize,dtype=np.float32)
    SEARCHLIMITY = np.zeros(origSize,dtype=np.float32)

    DX[0:Dx.shape[0],0:Dx.shape[1]] = Dx * scale_factor
    DY[0:Dy.shape[0],0:Dy.shape[1]] = Dy * scale_factor
    INTERPMASK[0:InterpMask.shape[0],0:InterpMask.shape[1]] = InterpMask
    CHIPSIZEX[0:ChipSizeX.shape[0],0:ChipSizeX.shape[1]] = ChipSizeX
    SEARCHLIMITX[0:SearchLimitX.shape[0],0:SearchLimitX.shape[1]] = SearchLimitX
    SEARCHLIMITY[0:SearchLimitY.shape[0],0:SearchLimitY.shape[1]] = SearchLimitY

    DX[noDataMask] = np.nan
    DY[noDataMask] = np.nan
    INTERPMASK[noDataMask] = 0
    CHIPSIZEX[noDataMask] = 0
    SEARCHLIMITX[noDataMask] = 0
    SEARCHLIMITY[noDataMask] = 0
    if SSM is not None:
        SSM[noDataMask] = False

    DX[SEARCHLIMITX == 0] = np.nan
    DY[SEARCHLIMITX == 0] = np.nan
    INTERPMASK[SEARCHLIMITX == 0] = 0
    CHIPSIZEX[SEARCHLIMITX == 0] = 0
    if SSM is not None:
        SSM[SEARCHLIMITX == 0] = False

    # Write MAT file
    import scipy.io as sio
    mat_dict = {'Dx':DX,
                'Dy':DY,
                'InterpMask':INTERPMASK,
                'ChipSizeX':CHIPSIZEX}
    sio.savemat('offset.mat', mat_dict)

    # Write geocoded file if grid provided
    if grid_location is not None:
        t1 = time.time()
        print("Write GeoTIFF Start!!!")

        # Create the GeoTiff
        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create("offset.tif", int(xGrid.shape[1]),
                                  int(xGrid.shape[0]), 4, gdal.GDT_Float32)
        outRaster.SetGeoTransform(tran)
        outRaster.SetProjection(proj)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(DX)
        outband.FlushCache()
        outband = outRaster.GetRasterBand(2)
        outband.WriteArray(DY)
        outband.FlushCache()
        outband = outRaster.GetRasterBand(3)
        outband.WriteArray(INTERPMASK)
        outband.FlushCache()
        outband = outRaster.GetRasterBand(4)
        outband.WriteArray(CHIPSIZEX)
        outband.FlushCache()
        del outRaster

        print("Write Outputs Done!!!")
        print(f"finished in {time.time()-t1:.3f} s")

    return None


def main():
    '''
    Main driver.
    Subroutine to retreive command line arguments and feed them to
    generateAutoriftProduct.
    No heavy lifting done here.
    '''
    inps = cmdLineParse()

    generateAutoriftProduct(**inps.__dict__)

    return None


if __name__ == '__main__':
    main()
