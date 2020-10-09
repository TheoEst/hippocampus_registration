# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:31:16 2020

@author: T_ESTIENNE
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math

def dice_loss(input, target):
    smooth = 1.
    target = target.float()
    input = input.float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))


def mean_dice_loss(input, target):
    
    assert input.shape[1] == 3
    assert target.shape[1] == 3
    
    dice = 0
    for i in range(1,3):
        dice += dice_loss(input[:, i, ...], target[:, i, ...])
    
    return dice /2 

def dice_metrics(mask, gt):
    '''
        Computes metrics based on the confusion matrix!
    '''
    lnot = np.logical_not
    land = np.logical_and

    true_positive = np.sum(land((mask), (gt)))
    false_positive = np.sum(land((mask), lnot(gt)))
    false_negative = np.sum(land(lnot(mask), (gt)))
    true_negative = np.sum(land(lnot(mask), lnot(gt)))

    M = np.array([[true_negative, false_negative],
                  [false_positive, true_positive]]).astype(np.float64)
    metrics = {}
    metrics['Sensitivity'] = M[1, 1] / (M[0, 1] + M[1, 1])
    metrics['Specificity'] = M[0, 0] / (M[0, 0] + M[1, 0])
    metrics['Dice'] = 2 * M[1, 1] / (M[1, 1] * 2 + M[1, 0] + M[0, 1])
    # metrics may be NaN if denominator is zero! use np.nanmean() while
    # computing average to ignore NaNs.

    return metrics['Dice']


def evalAllSample(mask, gt, moving_patients, reference_patients):

    batch_size = mask.shape[0]
    dice_dict = {}
    for batch in range(batch_size):
        
        # Label 1 : Head
        gt_ = gt[batch, 1, ...]
        mask_ = mask[batch, 1, ...] > 0.5
        head_dice = dice_metrics(mask_, gt_)
        
        # Label 2 : Tail
        gt_ = gt[batch, 2, ...]
        mask_ = mask[batch, 2, ...] > 0.5
        tail_dice = dice_metrics(mask_, gt_)

        dice_dict[moving_patients[batch]]  = {'Dice_head' : head_dice, 'Dice_tail': tail_dice,
                                              'Reference': reference_patients[batch]}
        
    return pd.DataFrame.from_dict(dice_dict, orient='index')



# From : https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/losses.py 

def ncc_loss(I, J):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, nb_feats, *vol_shape]
    """

    ndims = len(list(I.size())) - 2
    assert ndims == 3, "volumes should be 1 to 3 dimensions. found: %d" % ndims

    win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)   
    stride = (1,1,1)
    padding = (pad_no, pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return -1 * torch.mean(cc)



def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross
