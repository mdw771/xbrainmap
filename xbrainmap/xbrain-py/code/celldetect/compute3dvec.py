#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
Module for functions related to placing an iput 3D template at a fixed position in a bounding box.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

__author__ = "Eva Dyer"
__credits__ = "Mehdi Tondravi"

__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['roundno',
           'placeatom',
           'compute3dvec']


def roundno(no):
    
    """
    python rounds to the nearest even value. For exampl (12.5) returns 12. But in matlab
    round(12.5) returns 13. This function is to have the same behavior as in matlab code. Inconsistency
    is only if the number ends with ".5".
    """
    
    return int(no // 1 + ((no % 1) / 0.5) // 1)

def placeatom(vec, Lbox, which_loc, stacksz):
    
    """
    Parameters
    ----------
    vec : ndarray
        Nx1 array
    Lbox : int
        Lenght
    which_loc : int
        location to place atom
    stacksz : ndarry
        shape of the array (3D)
    
    Returns
    -------
    ndarray
    """
    tmp = np.zeros((stacksz))
    #Convert flat index to indices 
    r,c,z = np.unravel_index(which_loc, (stacksz)) 
    tmp[r, c, z] = 1
    
    # Increase every dimension by Lbox before, Lbox after each dimension and fill them with zeros
    tmp = np.lib.pad(tmp, ((Lbox, Lbox), (Lbox, Lbox), (Lbox, Lbox)), 'constant', constant_values=(0, 0))
    # get the indices of the nonzero element 
    center_loc = np.nonzero(tmp)
    Lbox_half = roundno(Lbox / 2)
    
    tmp[center_loc[0] - Lbox_half + 1:center_loc[0] + Lbox_half, \
            center_loc[1] - Lbox_half + 1:center_loc[1] + Lbox_half, \
            center_loc[2] - Lbox_half + 1:center_loc[2] + Lbox_half] = \
            np.reshape(vec, (Lbox, Lbox, Lbox))
    return(tmp)


def compute3dvec(vec,which_loc,Lbox,stacksz):
    
    """
    Parameters
    ----------
    vec : ndarray
        Nx1 array
    Lbox : int
        Lenght
    which_loc : int
        location to place atom
    stacksz : ndarry
        shape of the array (3D)
    
    Returns
    -------
    ndarray
    """
    
    tmp = placeatom(vec, Lbox, which_loc, stacksz)
    
    #delete the first Lbox R, C and Z 
    x,y,z = np.shape(tmp)
    tmp = tmp[Lbox:x, Lbox:y, Lbox:z]

    #delete the last Lbox R, C and Z
    x,y,z = np.shape(tmp)
    tmp = tmp[0:(x-Lbox), 0:(y-Lbox), 0:(z-Lbox)]
    return(tmp)
