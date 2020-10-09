# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:36:01 2020

@author: T_ESTIENNE
"""
import matplotlib.pyplot as plt
from hippocampus_registration import utils
import numpy as np

def plot_registration_results(gt_sample, pred_sample, batch, args, loss=None, 
                              deformed_mask=None):
    
    if args.deep_supervision:
        deformed_img = utils.to_numpy(args, pred_sample[0][2][-1])
        (deformable_grid, integrated_grid) = (utils.to_numpy(args, i) for i in pred_sample[0][:2])
    else:
        (deformable_grid, integrated_grid, deformed_img) = (utils.to_numpy(args, i) for i in pred_sample[0])
    
    loss = utils.to_numpy(args, loss)
    
    moving, reference = (utils.to_numpy(args, gt_sample[key]) for key in ['moving_irm', 'reference_irm'])
    
    if deformed_mask is None:
        reference_mask, moving_mask = None, None
    else:
        reference_mask, moving_mask = (utils.to_numpy(args, gt_sample[key]) for key in ['reference_mask', 'moving_mask'])
 
    fig_registration = plot_registration(moving, reference, 
                                         deformed_img, integrated_grid, batch,
                                         loss, moving_mask, reference_mask, 
                                         deformed_mask)
    
    return fig_registration


def plot_registration(moving, reference, 
                      deformed_img, grid, batch, loss=None,
                      moving_mask=None, reference_mask=None, deformed_mask=None):
    
    kwargs = {'cmap': 'gray'}

    titles = ['Target', 'Source', 'Deformed']
    
    if grid is not None:
        titles.append('Grid')
        
    if loss is not None:
        titles.append('Loss')
    
    if deformed_mask is not None:
        titles += ['Source Mask', 'Target Mask', 'Deformed_mask']
    
    nb_column = len(titles)

    fig, ax = plt.subplots(3, nb_column, gridspec_kw={'wspace': 0, 'hspace': 0.02,
                                              'top': 0.93, 'bottom': 0.01,
                                              'left': 0.01, 'right': 0.99})
    
    x_slice = int(moving.shape[2] // 2)
    y_slice = int(moving.shape[3] // 2)
    z_slice = int(moving.shape[4] // 2)

    modality = 0
    
    ax[0, 0].imshow(reference[batch, modality, x_slice, :, :], **kwargs)
    ax[1, 0].imshow(reference[batch, modality, :, y_slice, :], **kwargs)
    ax[2, 0].imshow(reference[batch, modality, :, :, z_slice], **kwargs)

    ax[0, 1].imshow(moving[batch, modality, x_slice, :, :], **kwargs)
    ax[1, 1].imshow(moving[batch, modality, :, y_slice, :], **kwargs)
    ax[2, 1].imshow(moving[batch, modality, :, :, z_slice], **kwargs)

    ax[0, 2].imshow(deformed_img[batch, modality, x_slice, :, :], **kwargs)
    ax[1, 2].imshow(deformed_img[batch, modality, :, y_slice, :], **kwargs)
    ax[2, 2].imshow(deformed_img[batch, modality, :, :, z_slice], **kwargs)
    
    column = 3
    
    if grid is not None:
        dz, dy, dx = (grid[batch, 0, :, :, :],
                      grid[batch, 1, :, :, :],
                      grid[batch, 2, :, :, :])
            
        ax[0, column].contour(dy[x_slice, ::-1, :], 50, alpha=0.90, linewidths=0.5)
        ax[0, column].contour(dz[x_slice, ::-1, :], 50, alpha=0.90, linewidths=0.5)
    
        ax[1, column].contour(dx[:, y_slice, :], 50, alpha=0.90, linewidths=0.5)
        ax[1, column].contour(dz[:, y_slice, :], 50, alpha=0.90, linewidths=0.5)
    
        ax[2, column].contour(dx[:, :, z_slice], 50, alpha=0.90, linewidths=0.5)
        ax[2, column].contour(dy[:, :, z_slice], 50, alpha=0.90, linewidths=0.5)
        
        column += 1
        
    if loss is not None:
        ax[0, column].imshow(loss[batch, 0, x_slice, :, :])
        ax[1, column].imshow(loss[batch, 0, :, y_slice, :])
        ax[2, column].imshow(loss[batch, 0, :, :, z_slice])
        column += 1
        
    if deformed_mask is not None:
        
        mask_kwargs = {'vmin':0, 'vmax':2}
        moving_mask = np.argmax(moving_mask, axis=1)
        reference_mask = np.argmax(reference_mask, axis=1)
        deformed_mask = np.argmax(deformed_mask, axis=1)
        
        ax[0, column].imshow(moving_mask[batch, x_slice, :, :], **mask_kwargs)
        ax[1, column].imshow(moving_mask[batch, :, y_slice, :], **mask_kwargs)
        ax[2, column].imshow(moving_mask[batch, :, :, z_slice], **mask_kwargs)
        column += 1
        
        ax[0, column].imshow(reference_mask[batch, x_slice, :, :], **mask_kwargs)
        ax[1, column].imshow(reference_mask[batch, :, y_slice, :], **mask_kwargs)
        ax[2, column].imshow(reference_mask[batch, :, :, z_slice], **mask_kwargs)
        column += 1
        
        ax[0, column].imshow(deformed_mask[batch, x_slice, :, :], **mask_kwargs)
        ax[1, column].imshow(deformed_mask[batch, :, y_slice, :], **mask_kwargs)
        ax[2, column].imshow(deformed_mask[batch, :, :, z_slice], **mask_kwargs)
        
    for j in range(nb_column):
        ax[0, j].set_title(titles[j])
        for i in range(3):
            ax[i, j].grid(False)
            ax[i, j].axis('off')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])



    fig.canvas.draw()

    return fig
