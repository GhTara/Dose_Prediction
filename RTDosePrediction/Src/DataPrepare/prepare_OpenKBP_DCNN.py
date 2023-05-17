# -*- encoding: utf-8 -*-
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import skimage.morphology
from scipy.ndimage import distance_transform_edt


# This function is adapted from OpenKBP official codes, https://github.com/ababier/open-kbp
def load_csv_file(file_name):
    """Load a file in one of the formats provided in the OpenKBP dataset
    :param file_name: the name of the file to be loaded
    :return: the file loaded
    """
    # Load the file as a csv
    loaded_file_df = pd.read_csv(file_name, index_col=0)

    # If the csv is voxel dimensions read it with numpy
    if 'voxel_dimensions.csv' in file_name:
        loaded_file = np.loadtxt(file_name)
    # Check if the data has any values
    elif loaded_file_df.isnull().values.any():
        # Then the data is a vector, which we assume is for a mask of ones
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:
        # Then the data is a matrix of indices and data points
        loaded_file = {'indices': np.array(loaded_file_df.index).squeeze(),
                       'data': np.array(loaded_file_df['data']).squeeze()}

    return loaded_file


# Transform numpy array(Z * H * W) to NITFI(nii) image
# Default image direction (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# Default origin (0.0, 0.0, 0.0)
def np2NITFI(image, spacing):
    image_nii = sitk.GetImageFromArray(image)
    image_nii.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    image_nii.SetSpacing(tuple(spacing))
    image_nii.SetOrigin((0.0, 0.0, 0.0))

    return image_nii


# Distance image proposed in https://doi.org/10.1088/1361-6560/aba87b
def get_distance_image(mask, spacing):
    mask_erode = skimage.morphology.binary_erosion(mask)
    surface = np.uint8(mask - mask_erode)
    distance = distance_transform_edt(np.logical_not(surface), sampling=[spacing[2], spacing[1], spacing[0]])
    distance[mask > 0] = -1 * distance[mask > 0]

    return distance.astype(np.float)


# Saving images to slices for 2D dose prediction methods
def save_slices(image, save_prefix,  list_slice_indexes, spacing):
    for index_ in list_slice_indexes:
        slice_ = image[index_:index_+1, :, :]
        slice_nii = np2NITFI(slice_, spacing)
        sitk.WriteImage(slice_nii, save_prefix + str(index_) + '.nii.gz')


if __name__ == '__main__':
    source_dir = '../../Data/open-kbp-master/provided-data'
    save_dir = '../../Data/OpenKBP_DCNN'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    list_patient_dirs = []
    for sub_dir in ['train-pats', 'validation-pats', 'test-pats']:
        for patient_id in os.listdir(source_dir + '/' + sub_dir):
            list_patient_dirs.append(source_dir + '/' + sub_dir + '/' + patient_id)

    for patient_dir in list_patient_dirs:
        # Make dir for each patient
        patient_id = patient_dir.split('/')[-1]
        save_patient_path = save_dir + '/' + patient_id
        if not os.path.exists(save_patient_path):
            os.mkdir(save_patient_path)

        # Spacing
        spacing = load_csv_file(patient_dir + '/voxel_dimensions.csv')

        # possible_dose_mask
        possible_dose_mask_csv = load_csv_file(patient_dir + '/possible_dose_mask.csv')
        possible_dose_mask = np.zeros((128, 128, 128), dtype=np.uint8)
        np.put(possible_dose_mask, possible_dose_mask_csv, np.uint8(1))
        possible_dose_mask = possible_dose_mask[:, :, ::-1].transpose([2, 0, 1])  # Data in OpenKBP dataset is (h, w, -z) or (y, x, -z)

        # Only slices receive dose are used
        list_slice_indexes = np.unique(np.where(np.sum(possible_dose_mask, axis=(1, 2)) > 5)).tolist()
        save_slices(possible_dose_mask,
                    save_prefix=save_patient_path + '/possible_dose_mask_',
                    list_slice_indexes=list_slice_indexes,
                    spacing=spacing)

        # CT
        CT_csv = load_csv_file(patient_dir + '/ct.csv')
        CT = np.zeros((128, 128, 128), dtype=np.int16)
        indices_ = np.int64(CT_csv['indices'])
        data_ = np.int16(CT_csv['data'])
        np.put(CT, indices_, data_)
        CT = CT - 1024

        CT = CT[:, :, ::-1].transpose([2, 0, 1])
        save_slices(CT,
                    save_prefix=save_patient_path + '/CT_',
                    list_slice_indexes=list_slice_indexes,
                    spacing=spacing)

        # Dose
        dose_csv = load_csv_file(patient_dir + '/dose.csv')
        dose = np.zeros((128, 128, 128), dtype=np.float32)
        indices_ = np.int64(dose_csv['indices'])
        data_ = np.float32(dose_csv['data'])
        np.put(dose, indices_, data_)

        dose = dose[:, :, ::-1].transpose([2, 0, 1])
        save_slices(dose,
                    save_prefix=save_patient_path + '/dose_',
                    list_slice_indexes=list_slice_indexes,
                    spacing=spacing)

        # OARs
        for structure_name in ['possible_dose_mask',
                               'Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid', 'Esophagus', 'Larynx',
                               'Mandible']:
            structure_csv_file = patient_dir + '/' + structure_name + '.csv'
            if os.path.exists(structure_csv_file):
                structure_csv = load_csv_file(structure_csv_file)
                structure = np.zeros((128, 128, 128), dtype=np.uint8)
                np.put(structure, structure_csv, np.uint8(1))

                structure = structure[:, :, ::-1].transpose([2, 0, 1])
                save_slices(structure,
                            save_prefix=save_patient_path + '/' + structure_name + '_',
                            list_slice_indexes=list_slice_indexes,
                            spacing=spacing)

        # PTVs and Distance images
        all_PTVs = np.zeros((128, 128, 128), np.float32)
        for structure_name in ['PTV70', 'PTV63', 'PTV56']:
            structure_csv_file = patient_dir + '/' + structure_name + '.csv'
            if os.path.exists(structure_csv_file):
                structure_csv = load_csv_file(structure_csv_file)
                structure = np.zeros((128, 128, 128), dtype=np.uint8)
                np.put(structure, structure_csv, np.uint8(1))

                structure = structure[:, :, ::-1].transpose([2, 0, 1])
                all_PTVs[structure > 0] = 1

                save_slices(structure,
                            save_prefix=save_patient_path + '/' + structure_name + '_',
                            list_slice_indexes=list_slice_indexes,
                            spacing=spacing)

        distnace_image = get_distance_image(all_PTVs, spacing)
        save_slices(distnace_image,
                    save_prefix=save_patient_path + '/distance_image_',
                    list_slice_indexes=list_slice_indexes,
                    spacing=spacing)

        print(patient_id + ' done !')
