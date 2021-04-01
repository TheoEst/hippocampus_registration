# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:01:35 2020

@author: T_ESTIENNE

Put zeros padding instead of constant padding
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

def load_sitk(path):
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    return img, array

def save_sitk(new_img, old_img, path):
    
    new_img.SetDirection(old_img.GetDirection())
    new_img.SetSpacing(old_img.GetSpacing())
    new_img.SetOrigin(old_img.GetOrigin())
    sitk.WriteImage(new_img, path)
    
    
def get_pad_index(array):
    
    
    x,y, z = array.shape
    mid_x, mid_y, mid_z = x //2, y//2, z//2
    
    sum_x = np.sum(array, axis=0)
    
    delta_y = sum_x[1:, :] - sum_x[0:-1, :]
    index_y = np.where(delta_y.sum(axis=1) == 0)[0]

    y_min = np.max(index_y[index_y < mid_y]) + 1 
    y_max = np.min(index_y[index_y > mid_y]) + 1
    
    delta_z = sum_x[:, 1:] - sum_x[:, 0:-1]
    index_z = np.where(delta_z.sum(axis=0) == 0)[0]
    
    z_min = np.max(index_z[index_z < mid_z]) + 1 
    z_max = np.min(index_z[index_z > mid_z]) + 1
    
    
    sum_y = np.sum(array, axis=1)
    delta_x = sum_y[1:, :] - sum_y[0:-1, :]
    index_x = np.where(delta_x.sum(axis=1) == 0)[0]
    
    x_min = np.max(index_x[index_x < mid_x]) + 1 
    x_max = np.min(index_x[index_x > mid_x]) + 1
    
    
    return x_min, x_max, y_min, y_max, z_min, z_max

main_path = './hippocampus_registration/data/'
main_path += 'L2R_Task4_HippocampusMRI/'
train = False
if train:
    folder = 'Training/'
else:
    folder = 'Testing/'

data_path = main_path + folder + 'img/'
save_path = main_path + folder + 'img_pad/'
patients = os.listdir(data_path)
plot=False

for patient in patients:
    
    img, array = load_sitk(data_path + patient)
    

    x_min, x_max, y_min, y_max, z_min, z_max = get_pad_index(array)

    if plot:
        plt.figure()
        plt.subplot(2,3,1)
        plt.imshow(array[32, ...], cmap='gray')
        
        plt.subplot(2,3,2)
        plt.imshow(array[:, 32, :], cmap='gray')
        
        plt.subplot(2,3,3)
        plt.imshow(array[:, :, 32], cmap='gray')
    

    array[:x_min, :, :] = 0
    array[x_max:, :,:] = 0
    array[:, :y_min, :] = 0
    array[:, y_max:,:] = 0
    array[:, :, :z_min] = 0
    array[:, :, z_max:] = 0
    
    new_img = sitk.GetImageFromArray(array)
    save_sitk(new_img, img, save_path + patient)
    if plot:
        plt.subplot(2,3,4)
        plt.imshow(array[32, ...], cmap='gray')
        
        plt.subplot(2,3,5)
        plt.imshow(array[:, 32, :], cmap='gray')
        
        plt.subplot(2,3,6)
        plt.imshow(array[:, :, 32], cmap='gray')
    
