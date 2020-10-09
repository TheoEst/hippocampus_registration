# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:43:36 2020

@author: T_ESTIENNE
"""
import scipy
import scipy.ndimage
import numpy as np
import random
from abdominal_registration import rotate
from threadpoolctl import threadpool_limits

def RandomTranslation(max_translation=30, transform_matrix=None):
    
    translation = [random.randint(-max_translation, max_translation) for i in range(3)]
    
    return getTranslationMatrix(translation, transform_matrix)
    
def RandomRotation(theta_max=20, only_one_rotation=True,
                   transform_matrix=None):
    
    theta = [random.uniform(-theta_max, theta_max) for i in range(3)]
    
    if only_one_rotation:
        axis = random.sample([0,1,2], 2)
        for i in axis:
            theta[i] = 0

    return getRotationMatrix(theta, transform_matrix)

def RandomZoom(zoom_max=0.2, isotropique_zoom=False,
               transform_matrix=None):
    
    if isotropique_zoom:
        zoom = random.uniform(1 - zoom_max, 1 + zoom_max)
        zoom = 3*[zoom]
    else:
        zoom = [random.uniform(1 - zoom_max, 1 + zoom_max) for i in range(3)]

    return getZoomMatrix(zoom, transform_matrix)

def getTranslationMatrix(translation, transform_matrix=None):
    '''
        2D translation on the axis (0, 1). 
        Axis 3 is the modality axis
        tx: Width shift.
        ty: Heigh shift.
    
    '''

    shift_matrix = np.array([[1, 0, 0, translation[0]],
                            [0, 1, 0, translation[1]],
                            [0, 0, 1, translation[2]],
                            [0, 0, 0, 1]])

    if transform_matrix is None:
        transform_matrix = shift_matrix
    else:
        transform_matrix = np.dot(transform_matrix, shift_matrix)
            
    return transform_matrix

def getZoomMatrix(zoom, transform_matrix=None):
    '''
        Affine Zoom in 2D
        zx: Zoom in x direction.
        zy: Zoom in y direction
    '''
    zoom_matrix = np.array([[zoom[0], 0, 0, 0],
                            [0, zoom[1], 0, 0],
                            [0, 0, zoom[2], 0],
                            [0, 0, 0, 1]])
    if transform_matrix is None:
        transform_matrix = zoom_matrix
    else:
        transform_matrix = np.dot(transform_matrix, zoom_matrix)
            
    return transform_matrix

def getRotationMatrix(theta, transform_matrix=None):
    '''
        2D rotation on the axis (0, 1). 
        Axis 3 is the modality axis
        theta: Rotation angle in degrees.
    '''
    theta = np.deg2rad(theta)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotate.create_matrix_rotation_3d(theta[0],
                                                                 theta[1],
                                                                 theta[2])
    
    if transform_matrix is None:
        transform_matrix = rotation_matrix
    else:
        transform_matrix = np.dot(transform_matrix, rotation_matrix)
    
    return transform_matrix

def create_coordinate_mesh(shape, crop_shape=None):
    '''
        output of 4*shape  
    '''
    if crop_shape is not None:
        (depth_min, depth_max, height_min, height_max, 
             width_min, width_max) = crop_shape
    else:
        depth_min, height_min, width_min = 0, 0, 0
        depth_max, height_max, width_max = shape
        
    z = np.arange(depth_min, depth_max)
    y = np.arange(height_min, height_max)
    x = np.arange(width_min, width_max)
    
    coords = np.array(np.meshgrid(z, y, x, indexing='ij')).astype(float)
    # Stack for homogeneous coordinates
    
    return np.vstack([coords, np.ones((1,*coords.shape[1:]))])

def apply_affine_transform(x, seg=None, transform_matrix=None, crop_shape=None, 
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 4D numpy array, single image, multimodalities (Modality*H*W)
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation
    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    if transform_matrix is not None:
        
        channels, h, w, d = x.shape
                
        transform_matrix = transform_matrix_offset_center(transform_matrix, 
                                                          h, w, d)
        coords = create_coordinate_mesh(x.shape[1:], crop_shape)
        
        # Multiplication between coords and trasnform matrix
        with threadpool_limits(limits=1, user_api='blas'):
            
            trf_coords = coords.reshape(coords.shape[0], -1)
            trf_coords = np.matmul(transform_matrix, trf_coords)
            trf_coords = trf_coords.reshape(*coords.shape)
        
        trf_coords = trf_coords[:-1, :, :, :] 

        # Interpolation
        res = [ scipy.ndimage.map_coordinates(x[channel, ...], trf_coords,
                                              order=order, mode=fill_mode, cval=cval)  for channel in range(channels)]
        x = np.stack(res, axis=0)
        x[x < 1e-3] = 0
        
        if seg is not None:
            
            labels = seg.shape[0]
            res = [scipy.ndimage.map_coordinates(seg[label, ...], trf_coords,
                                                                order=order, mode=fill_mode, cval=cval) for label in range(labels)]
            
            seg = np.stack(res, axis=0)
            seg[seg > 0.5] = 1
            seg[seg < 0.5] = 0
            
        
    return x, seg

def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], 
                              [0, 1, 0, o_y], 
                              [0, 0, 1, o_z], 
                              [0, 0, 0, 1]])
    
    reset_matrix = np.array([[1, 0, 0, -o_x], 
                             [0, 1, 0, -o_y], 
                             [0, 0, 1, -o_z], 
                             [0, 0, 0, 1]])
    
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    
    return transform_matrix
