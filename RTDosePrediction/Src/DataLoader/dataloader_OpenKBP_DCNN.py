# -*- encoding: utf-8 -*-
import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2

from DataAugmentation.augmentation_OpenKBP_DCNN import \
    random_flip_2d,  random_rotate_around_z_axis, random_translate, to_tensor

"""
Output images are always C*H*W
"""


def read_data(patient_dir, slice_index):
    dict_images = {}
    list_structures = ['CT',
                       'possible_dose_mask',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'distance_image',
                       'Brainstem',
                       'SpinalCord',
                       'RightParotid',
                       'LeftParotid',
                       'Esophagus',
                       'Larynx',
                       'Mandible',
                       'dose']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '_' + str(slice_index) + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        elif structure_name in ['dose', 'distance_image']:
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128), np.uint8)

    return dict_images


def pre_processing(dict_images):
    # PTVs
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']

    # OARs
    list_OAR_names = ['Brainstem',
                      'SpinalCord',
                      'RightParotid',
                      'LeftParotid',
                      'Esophagus',
                      'Larynx',
                      'Mandible'
                      ]
    OAR_all = np.zeros((1, 128, 128), np.uint8)
    for OAR_i in range(7):
        OAR = dict_images[list_OAR_names[OAR_i]]
        OAR_all[OAR > 0] = OAR_i + 1

    # CT image
    CT = dict_images['CT']
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.

    # Dose
    dose = dict_images['dose'] / 70.

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images['possible_dose_mask']

    # Distance image proposed in https://doi.org/10.1088/1361-6560/aba87b
    distance_image = dict_images['distance_image'] / 50.

    list_images = [np.concatenate((PTVs, OAR_all, CT, distance_image), axis=0),  # Input
                   dose,  # Label
                   possible_dose_mask]
    return list_images


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_2d(list_images, list_axis=[1], p=1.0)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angle=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_boder_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,
                                   roi_mask=list_images[2][0, :, :],  # the possible dose mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])

    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class MyDataset(data.Dataset):
    def __init__(self, num_samples_per_epoch, phase):
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {'train': train_transform, 'val': val_transform}

        self.list_case_id = {'train': ['../../Data/OpenKBP_DCNN/pt_' + str(i) for i in range(1, 201)],
                             'val': ['../../Data/OpenKBP_DCNN/pt_' + str(i) for i in range(201, 241)]}[phase]

        random.shuffle(self.list_case_id)
        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            case_id = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            case_id = self.list_case_id[new_index_]

        # Randomly pick a slice as input
        list_files = os.listdir(case_id)
        target_slice = int(random.sample(list_files, 1)[0].split('_')[-1].split('.')[0])

        dict_images = read_data(case_id,  slice_index=target_slice)
        list_images = pre_processing(dict_images)

        list_images = self.transform[self.phase](list_images)

        return list_images

    def __len__(self):
        return self.num_samples_per_epoch


def get_loader(train_bs=1, val_bs=1, train_num_samples_per_epoch=1, val_num_samples_per_epoch=1, num_works=0):
    train_dataset = MyDataset(num_samples_per_epoch=train_num_samples_per_epoch, phase='train')
    val_dataset = MyDataset(num_samples_per_epoch=val_num_samples_per_epoch, phase='val')

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=False)

    return train_loader, val_loader
