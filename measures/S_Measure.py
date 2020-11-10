"""
# > Modules for computing Structure measure (S-Measure)  
# Maintainer: https://github.com/xahidbuffon
"""
from __future__ import division
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass


def get_SMeasure(smap, gt):
    """
       - Given a saliency map (smap) and ground truth (gt)
       - Return Structure measure
         * Structure measure: A new way to evaluate foreground maps 
           [Deng-Ping Fan et. al - ICCV 2017] 
         //see https://github.com/DengPingFan/S-measure
    """
    assert (smap.shape==gt.shape)
    mu = np.mean(gt) # ratio of foreground area in gt

    def pixel_sim_score(a, b):
        # see how many b==1 (gt) pixels are correct in a 
        x = a[b==1]
        mu_x, sig_x = np.mean(x), np.std(x)
        # score based on the mean and std of accuracy 
        return 2.0 * mu_x/(mu_x**2 + 1.0 + sig_x + 1e-8)

    def S_object():
        # foreground similarity
        smap_fg = np.logical_and(smap, gt)
        O_FG = pixel_sim_score(smap_fg, gt)
        # background similarity
        smap_bg = np.logical_and(1-smap, 1-gt)
        O_BG = pixel_sim_score(smap_bg, 1-gt)
        # return combined score 
        return mu * O_FG + (1 - mu) * O_BG

    def S_region():
        # find the centroid of the gt
        xc, yc = map(int, center_of_mass(gt))
        # divide gt into 4 regions
        gt1, w1, gt2, w2, gt3, w3, gt4, w4 = get_quad_mask(xc, yc, gt)
        # divide smap into 4 regions
        smap1, _, smap2, _, smap3, _, smap4, _ = get_quad_mask(xc, yc, smap)
        # compute the ssim score for each regions
        Sr1 = get_SSIM_bin(smap1, gt1); Sr2 = get_SSIM_bin(smap2, gt2);
        Sr3 = get_SSIM_bin(smap3, gt3); Sr4 = get_SSIM_bin(smap4, gt4);
        # return weighted sum
        return w1 * Sr1 + w2 * Sr2 + w3 * Sr3 + w4 * Sr4

    def get_quad_mask(xc, yc, mask):
        # divide mask into 4 regions a given centroid (x, y)
        imH, imW = mask.shape; area = imW * imH;
        # 4 regions R1-R4: weights are proportional to their area
        R1 = mask[0:yc, 0:xc];     w1 = (1.0 * xc * yc)/area;
        R2 = mask[0:yc, xc:imW];   w2 = (1.0 * (imW - xc) * yc)/area;
        R3 = mask[yc:imH, 0:xc];   w3 = (1.0 * xc * (imH - yc))/area; 
        R4 = mask[yc:imH, xc:imW]; w4 = (1.0 - w1 - w2 - w3);
        return R1, w1, R2, w2, R3, w3, R4, w4

    def get_SSIM_bin(X, Y):
        N = np.size(Y)
        X = X.astype(np.float32); mu_x = np.mean(X)
        Y = Y.astype(np.float32); mu_y = np.mean(Y)
        #Compute the variance of SM,GT   
        sigma_x = np.sum((X - mu_x)**2)/(N - 1 + 1e-8) 
        sigma_y = np.sum((Y - mu_y)**2)/(N - 1 + 1e-8)
        #Compute the covariance between SM and GT
        sigma_xy = np.sum((X - mu_x)*(Y - mu_y))/(N - 1 + 1e-8)
        alpha = 4 * mu_x * mu_x * sigma_xy;
        beta = (mu_x**2 + mu_y**2) * (sigma_x + sigma_y);
        ssim = alpha/(beta + 1e-8)
        if alpha!=0: return ssim
        else: return 1 if (alpha==0 and beta==0) else 0
    #######################################################
    if mu==0 or mu==1: # if gt is completely black or white
        mu_s = np.mean(smap) # only get the intersection
        S = 1-mu_s if mu==0 else mu_s
    else:
        alpha = 0.5
        S = alpha * S_object() + (1-alpha) * S_region()           
    return S if S>0 else 0


