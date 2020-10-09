    # -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:40:07 2020

@author: T_ESTIENNE
"""
import numpy as np
from hippocampus_registration import affine_transform
import random 

def crop(array, size):

    depth_min, depth_max, height_min, height_max, width_min, width_max = size

    if len(array.shape) == 3:
        crop_array = array[depth_min:depth_max,
                           height_min:height_max,
                           width_min:width_max,
                           ]
    elif len(array.shape) == 4:
        crop_array = array[:, depth_min:depth_max,
                           height_min:height_max,
                           width_min:width_max,
                           ]
    else:
        print(array.shape)
        raise ValueError

    return crop_array


class Crop(object):

    def __init__(self, output_size, dim=3, do_affine=False):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = dim * (output_size,)
        else:
            assert len(output_size) == dim
            self.output_size = output_size
        self.do_affine = do_affine
        
    def __call__(self, sample):
        pass

def center_crop_indices(img, output_size):
    
    _, depth, height, width = img.shape
    
    if depth == output_size[0]:
        depth_min = 0
        depth_max = depth
    else:
        depth_min = int((depth - output_size[0])/2)
        depth_max = depth_min + output_size[0]
    
    if height == output_size[1]:
        height_min = 0
        height_max = height
    else:
        height_min = int((height - output_size[1])/2)
        height_max = output_size[1] + height_min
    
    if width == output_size[2]:
        width_min = 0
        width_max = width
    else:
        width_min = int((width - output_size[2])/2)
        width_max = output_size[2] + width_min
    
    return (depth_min, depth_max,
            height_min, height_max,
            width_min, width_max)

class CenterCrop(Crop):
    """Crop randomly the image in a sample centerd on the tumor

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
        dim (int) : Dimension of the input volumes (2D or 3D)
    """

    def __init__(self, output_size, dim=3, do_affine=False):
        super(CenterCrop, self).__init__(output_size, dim, do_affine)

    def __call__(self, sample):
        
        new_sample = []
        for (irm, mask) in sample:
            
            crop_shape = center_crop_indices(irm, self.output_size)
            
            if self.do_affine:
                new_sample.append((irm, mask, crop_shape))
            else:
                new_irm = crop(irm, crop_shape)
                new_mask = None if mask is None else crop(mask, crop_shape)
                new_sample.append((new_irm, new_mask, None))

        return new_sample


class RandomCrop(Crop):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

    def __init__(self, output_size, dim=3, do_affine=False):
        super(RandomCrop, self).__init__(output_size, dim, do_affine)

    def __call__(self, sample):
        
        irm = sample[0][0]
        
        _, depth, height, width = irm.shape
        
        i = random.randint(0, depth - self.output_size[0])
        j = random.randint(0, height - self.output_size[1])
        k = random.randint(0, width - self.output_size[2])
    
        crop_shape = (i, i + self.output_size[0],
                j, j + self.output_size[1],
                k, k + self.output_size[1])
        
        new_sample = []
        for (irm, mask) in sample:

            if self.do_affine:
                new_sample.append((irm, mask, crop_shape))
            else:
                new_irm = crop(irm, crop_shape)
                new_mask = None if mask is None else crop(mask, crop_shape)
                new_sample.append((new_irm, new_mask, None))

        return new_sample


def normalize(img):
    
    mean = np.mean(img[img > 0])
    std = np.std(img[img > 0])

    img = (img - mean) / std
    img = np.clip(img, -5, 5)

    mini = np.min(img)
    maxi = np.max(img)

    array = (img - mini) / (maxi - mini)

    return array

class Normalize(object):
    """ Normalize the dicom image in sample. The dicom image must be a Tensor"""

    def __call__(self, sample):
        
        new_sample = []
        for (irm, mask, crop_shape) in sample:
            
            new_irm = normalize(irm)
            new_sample.append((new_irm, mask, crop_shape))


        return new_sample


class AxialFlip(object):

    def __call__(self, sample):

        choice_x = random.randint(0, 1)
        choice_y = random.randint(0, 1)
        choice_z = random.randint(0, 1)
        
        new_sample = []
        
        for (irm, mask, crop_shape) in sample:
            new_irm = self.axialflip(irm, choice_x, choice_y, choice_z)
            if mask is None:
                new_mask = None
            else:
                new_mask = self.axialflip(mask, choice_x, choice_y, choice_z)
            new_sample.append((new_irm, new_mask, crop_shape))


        return new_sample

    def axialflip(self, array, choice_x, choice_y, choice_z):

        ndim = len(array.shape)

        if choice_x == 1:
            if ndim == 3:
                array = array[:, :, ::-1]
            elif ndim == 4:
                array = array[:, :, :, ::-1]
            else:
                raise ValueError

        if choice_y == 1:
            if ndim == 3:
                array = array[:, ::-1, :]
            elif ndim == 4:
                array = array[:, :, ::-1, :]
            else:
                raise ValueError
                
        if choice_z == 1:
            if ndim == 3:
                array = array[::-1, ...]
            elif ndim == 4:
                array = array[:, ::-1, :, :]
            else:
                raise ValueError

        return np.ascontiguousarray(array)


class GaussianNoise(object):

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, sample):

        new_sample = []
        
        for irm, mask, crop_shape in sample:

            new_irm = irm + np.random.randn(*irm.shape).astype(irm.dtype) * \
                    self.std + self.mean
            new_sample.append((new_irm, mask, crop_shape))
            
        return new_sample


class RandomRotation90(object):
    '''
        Taken from augment_rot90 from MIC-DKFZ/batchgenerators
        https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/spatial_transformations.py
    '''

    def __init__(self, num_rot=(1, 2, 3, 4), axes=(0, 1, 2)):

        self.num_rot = num_rot
        self.axes = axes

    def __call__(self, sample):

        num_rot = random.choice(self.num_rot)
        axes = random.sample(self.axes, 2)
        axes.sort()
        axes = [i + 1 for i in axes] # img has shap of lenght 4
        
        def f(img):
            return np.ascontiguousarray(np.rot90(img, num_rot, axes))
        
        new_sample = []
        for irm, mask, crop_shape in sample:
            new_irm = f(irm)
            new_mask = None if mask is None else f(mask)
            new_sample.append((new_irm, new_mask, crop_shape))
        return new_sample


class AffineTransform(object):
    
    def __init__(self, data_aug_kwargs):

        self.data_aug_kwargs = data_aug_kwargs
        
        self.theta = data_aug_kwargs['theta']
        self.only_one_rotation = data_aug_kwargs['only_one_rotation']
        self.max_translation = data_aug_kwargs['max_translation']
        self.max_zoom = data_aug_kwargs['max_zoom']
        self.isotropique_zoom = data_aug_kwargs['isotropique_zoom']
        
    def __call__(self, sample):
        
        transform_matrix = np.eye(4)

        if self.theta > 0:
            transform_matrix = affine_transform.RandomRotation(self.theta, 
                                                               self.only_one_rotation)

        if self.max_translation > 0:
            transform_matrix = affine_transform.RandomTranslation(self.max_translation,
                                                                  transform_matrix)

        if self.max_zoom > 0:
            transform_matrix = affine_transform.RandomZoom(self.max_zoom,
                                                           self.isotropique_zoom,
                                                           transform_matrix)
        new_sample = []
        
        for irm, mask, crop_shape in sample:
            new_irm, new_mask = affine_transform.apply_affine_transform(irm,
                                                                        mask, 
                                                                        transform_matrix,
                                                                        crop_shape)
            new_sample.append((new_irm, new_mask, crop_shape))
            
        return new_sample

