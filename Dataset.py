# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:26:48 2020

@author: T_ESTIENNE
"""
import os
import numpy as np
import time
import torch.utils.data as data
import SimpleITK as sitk
import random
import nibabel.freesurfer.mghformat as mgh
import pandas as pd 


end = '.nii.gz'
seg_name = '_seg'

def oasis2brats(array):
    '''
        From oasis space to BRATS space
    '''
    tab = np.transpose(array, (1, 2, 0) )
    tab = tab[::-1, ::-1, ::-1]
    tab = tab[50:-51, 8:-8, 8:-8]
    return tab

def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def load_mgh(path):
    return mgh.load(path).get_data()

def load_dataset(dataset_path, cohort='learn2reg',
                 test=False):
    
    if cohort == 'learn2reg':
        train_set = np.loadtxt(dataset_path + 'L2R_file_id_train.txt', dtype=str)# 200
        val_or_test = 'test' if test else 'val'
        val_set = np.loadtxt(dataset_path + 'L2R_file_id_{}.txt'.format(val_or_test), dtype=str)# shape 60
        val_pairs = pd.read_csv(dataset_path + 'L2R_pairs_{}.csv'.format(val_or_test)).values# shape 60*2

        return train_set, val_set, val_pairs
    
    else:
        train_set = np.loadtxt(dataset_path + cohort + '_train.txt', dtype=str)
        val_set = np.loadtxt(dataset_path + cohort + '_val.txt', dtype=str)
        
        return train_set, val_set


class RegistrationDataset(data.Dataset):
    """Dicom dataset."""

    def __init__(self, files_list, folder, transform=None, pairs=None,
                 verbosity=False):
        """
        Args:
            samples_list (list): List tuples of format (dicom,tumor)

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(RegistrationDataset, self).__init__()
        
        self.folder = folder
        self.files_list = files_list
        self.pairs = pairs
        self.transform = transform
        self.verbosity = verbosity
        
    def __len__(self):
        return len(self.files_list)

    def load_hippocampus(self, patient):
        '''
            Load a dicom, tumor or ring
        '''
        start = time.time()
        
        patient = int(patient)
        irm_path = self.folder + 'img_pad/hippocampus_{:03d}.nii.gz'.format(patient)
        array = load_sitk(irm_path)[np.newaxis, ...].astype(float)# Add 1 dimension
        
        mask_path = self.folder + 'label/hippocampus_{:03d}.nii.gz'.format(patient)
        mask = load_sitk(mask_path)
        
        label = 3
        mask = mask.astype(np.int16)
        mask = np.rollaxis(np.eye(label, dtype=np.uint8)[mask], -1, 0)
        
        stop = time.time()

        return array, mask, stop - start
    
    def __getitem__(self, idx):
        
        if self.pairs is not None:
            reference_patient, moving_patient = self.pairs[idx, :]
        else:    
            moving_patient = self.files_list[idx]
            reference_patient = random.choice(self.files_list)
        

        start = time.time()
        moving_irm, moving_mask, time_load = self.load_hippocampus(moving_patient)
        reference_irm, reference_mask, _ = self.load_hippocampus(reference_patient)

        sample = [(reference_irm, reference_mask), (moving_irm, moving_mask)]
        
        if self.transform:
            start_transform = time.time()
            [(reference_irm, reference_mask,_), 
             (moving_irm, moving_mask, _)] = self.transform(sample)
            time_transform = time.time() - start_transform
        
        if self.verbosity:
            print('Sample import = {}'.format(2*time_load))
            print('Sample Transformation = {}'.format(time_transform))
            print('Total time for sample = {}'.format(time.time() - start))
            
            
        bbox = self.create_bounding_box(moving_irm, reference_irm)
        
        sample = {'reference_irm': reference_irm, 'reference_mask': reference_mask,
                  'moving_irm': moving_irm, 'moving_mask': moving_mask,
                  'bbox' : bbox,
                  'reference_patient': reference_patient, 
                  'moving_patient': moving_patient}

        return sample

    def create_bounding_box(self, moving, reference):
        '''
        find the position of the mri and create a bounding_box

        Parameters
        ----------
        irm : shape 1*W*H*D

        Returns
        -------
        bounding_box : shape 2*W*H*S

        '''
        shape = moving.shape[1:]
        bbox = np.zeros((2, *shape), dtype=moving.dtype)
        
        bbox[0, ...] = np.logical_or( moving == 0, reference==0)
        bbox[1, ...] = np.logical_and(moving>0, reference >0)
        
        return bbox


class OasisDataset(data.Dataset):
    """Dicom dataset."""

    def __init__(self, files_list, folder, transform=None, verbosity=False,
                 validation=False
                 ):
        """
        Args:
            samples_list (list): List tuples of format (dicom,tumor)

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(OasisDataset, self).__init__()
        
        self.files_list = files_list
        self.oasis_path = os.path.join(folder, 'oasis')
        self.validation = validation
        self.verbosity = verbosity
        self.transform = transform

    def __len__(self):
        return len(self.files_list)
    
        
    def load_oasis(self, patient):
        
        start = time.time()

        array_path = os.path.join(self.oasis_path, 'irm/' + patient)
            
        array = load_mgh(array_path)

        array = oasis2brats(array)[np.newaxis, ::-1, :, :]
   
        stop = time.time()

        return array, stop - start
    
    def __getitem__(self, idx):

        moving_patient = self.files_list[idx]
        if self.validation:
            reference_patient = self.files_list[ (idx+1) % len(self)]
        else:
            reference_patient = random.choice(self.files_list)

        start = time.time()
        moving_irm, time_load = self.load_oasis(moving_patient)
        reference_irm, _ = self.load_oasis(reference_patient)

        sample = [(reference_irm, None), (moving_irm, None)]

        if self.transform:
            start_transform = time.time()
            [(reference_irm, _,_), 
             (moving_irm, _, _)] = self.transform(sample)
            time_transform = time.time() - start_transform
        
        if self.verbosity:
            print('Sample import = {}'.format(2*time_load))
            print('Sample Transformation = {}'.format(time_transform))
            print('Total time for sample = {}'.format(time.time() - start))
            
               
        sample = {'reference_irm': reference_irm, 'moving_irm': moving_irm,
                  'reference_patient': reference_patient, 'moving_patient': moving_patient}

        return sample


    
def init_datasets(transformation, val_transformation, args):
    
    args.folder = args.main_path + 'data/'
    
    args.folder += 'L2R_Task4_HippocampusMRI/'
    args.folder += 'Testing/' if args.test else 'Training/'

    (files_train, files_val, pairs_val) = load_dataset(args.dataset_path, 
                                                       test=args.test)
                                      
    if args.debug:
        files_train = files_train[:2*args.batch_size+1]
        files_val = files_val[:2*args.batch_size+1]

    if args.merge_train_val:
        files_train = np.concatenate([files_train, files_val])
        
    dataset_kwargs = {'verbosity': args.verbosity,
                      'folder' : args.folder}

    val_Dataset = RegistrationDataset(files_val,
                                      pairs=pairs_val,
                                      transform=val_transformation,
                                      **dataset_kwargs)
    
    train_Dataset = RegistrationDataset(files_train,
                                        transform=transformation,
                                        **dataset_kwargs)
     
    return train_Dataset, val_Dataset


def init_pretrain_datasets(transformation, val_transformation, args):
    
    args.folder = args.main_path + 'data/'

    files_train, files_val = load_dataset(args.dataset_path, 'oasis')
    
    if args.debug:
        files_train = files_train[:2*args.batch_size+1]
        files_val = files_val[:2*args.batch_size+1]

    dataset_kwargs = {'verbosity': args.verbosity,
                      'folder' : args.folder}
    
    val_Dataset = OasisDataset(files_val, transform=val_transformation,
                                validation=True, **dataset_kwargs)
    
    train_Dataset = OasisDataset(files_train, 
                                 transform=transformation,
                                 **dataset_kwargs)

    return train_Dataset, val_Dataset

