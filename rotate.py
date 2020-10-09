# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:42:45 2020

@author: T_ESTIENNE

https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/rand_rotation.py

"""
import numpy as np

def create_matrix_rotation_x_3d(angle):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])

    return rotation_x


def create_matrix_rotation_y_3d(angle):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])

    return rotation_y


def create_matrix_rotation_z_3d(angle):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])

    return rotation_z


def create_matrix_rotation_3d(angle_x, angle_y, angle_z):

    transform_x = create_matrix_rotation_x_3d(angle_x)
    transform_y = create_matrix_rotation_y_3d(angle_y)
    transform_z = create_matrix_rotation_z_3d(angle_z)
    
    transform = np.dot(transform_z, np.dot(transform_x, transform_y))
    
    return transform


    