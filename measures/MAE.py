"""
# > Module for computing Mean Absolute Error (MAE)  
# Maintainer: https://github.com/xahidbuffon
"""
from __future__ import division
import os
import numpy as np


def get_MAE(smap, gt):
    """
       - Given a saliency map (smap) and ground truth (gt)
       - Return MAE - Mean Absolute Error  
    """
    assert (smap.shape==gt.shape)
    #mae = np.mean(np.absolute((gt.astype("float")-smap.astype("float"))))
    mae = np.mean(np.logical_xor(smap, gt)) # same result
    return mae


