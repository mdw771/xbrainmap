#!/usr/bin/env python

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Converts reconstructed tiff image file into hdf5 file.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import h5py
import numpy as np
from skimage.io import imread
from glob import glob
import os.path
from segmentation_param import *
from mpi4py import MPI
import time

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['tiff_to_hdf5_files']

def tiff_to_hdf5_files():
    """
    Create a directory to hold the hdf file at same level as tiff files directory. This directory name
    is the same as tiff directory name plus "_hdf". Create an hdf5 file with the name made of 
    the concatenation of the first tiff file and last tiff file. Create an hdf data set name with the 
    same name as the directory name of the tiff files.
    As an example if the tiff files are located at, "/path/Downloads/eva_block" directory,
    the first tiff name is "data_00860.tiff.tif" and the last tiff file name is "data_01139.tiff.tif" 
    then the hdf file created is:
    /path/Downloads/eva_block_hdf/data_00860.tiff_data_01139.tiff.hdf5
    and hdf5 data set name is "eva_block".
    
    Input: Tiff files location is specified in the segmentation_param.py file.
    
    Output: HDF5 files location is specified in the segmentation_param.py file.
    
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    
    start_time = int(time.time())
    files = sorted(glob(tiff_files_location + '/*.tif*'))
    parent_dir, tiff_dir = os.path.split(tiff_files_location)
    hdf_dir = parent_dir + '/' + tiff_dir + '_hdf'

    # Make a list with files to be processed by each process/rank.
    files_per_rank = int(len(files)/size)


    # Remove all *.hdf5 files from previous runs. Create directory if does not exist
    if rank == 0:
        if os.path.exists(hdf_dir):
           hdf5_files =  glob(hdf_dir + '/*.hdf5')
           for file in hdf5_files:
               os.remove(file)
        if not os.path.exists(hdf_dir):
            os.mkdir(hdf_dir)

    data_shape = imread(files[0]).shape
    data_type = imread(files[0]).dtype
    hdf_file_name = hdf_dir + '/bacon.hdf5'
    hdf_file = h5py.File(hdf_file_name, 'w+')
    data_set_name = 'image'
    data_set = hdf_file.create_dataset(data_set_name, (len(files), data_shape[0], data_shape[1]), data_type)
    comm.Barrier()
    

    for idx in range(rank*len(files_per_rank),(rank+1)*len(files_for_rank)):
        data_set[idx, :, :] = imread(idx)
    
    print("data shape is", data_set.shape)
    hdf_file.close()
    end_time = int(time.time())
    exec_time = end_time - start_time
    print("Done dividing tiff files, rank is %d, size is %d, name is %s, exec time is %d sec" % (rank, size, name, exec_time))


if __name__ == '__main__':
    tiff_to_hdf5_files()

