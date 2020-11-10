"""
# > Modules for computing F-Measure
# Maintainer: https://github.com/xahidbuffon
"""
import os
import ntpath
import numpy as np


def get_PR_uint8(smap, gt):
    """
       - Given a saliency map (smap) and ground truth (gt)
           - smap is uint8 (0, 255) and gt is binary (0/1)
       - Return P: Precision and R: Recall 
           - P = TP/(TP+FP) and R = TP/(TP+FN)
           - 0:255 bins of thresholds
           - return array (255,) 
    """
    if np.max(gt)>1: 
        gt = gt/255.0 # force binary
        gt = (gt > 0.5).astype(np.int32)
    N_tp = np.sum(gt)
    # bin the prediction and target in [0, 255]
    union_hist, _ =  np.histogram(smap, range(256))
    hit_hist, _  = np.histogram(smap[gt==1], range(256))
    miss_hist, _ = np.histogram(smap[gt==0], range(256))
    # take cumulative scores
    hit_hist = np.cumsum(np.flipud(hit_hist))
    miss_hist = np.cumsum(np.flipud(miss_hist))
    union_hist = np.cumsum(np.flipud(union_hist)) + N_tp - hit_hist
    # make the calculations
    precisions = hit_hist / (hit_hist + miss_hist + 1e-8) 
    recalls = hit_hist / (N_tp + 1e-8)
    ious = hit_hist / (union_hist + 1e-8)
    return precisions, recalls, ious 


def get_wFMeasure(P, R, beta_2=0.3):
    """
       - Given precision P and recall R
       - Return weighted F_measure (max) = (1+beta^2)* P * R / (beta^2 * P + R)
         beta^2 = 0.3 as per the SOTA methods
    """
    return ((1+beta_2) * P * R) / (beta_2*P + R + 1e-8)


