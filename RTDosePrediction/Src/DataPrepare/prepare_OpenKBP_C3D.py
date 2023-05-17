# -*- encoding: utf-8 -*-
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os


# This function is adapted from OpenKBP official codes, https://github.com/ababier/open-kbp
def load_csv_file(file_name):
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


if __name__ == '__main__':
    source_dir = '../../Data/open-kbp-master/provided-data'
    save_dir = '../../Data/OpenKBP_C3D'

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

        # CT
        CT_csv = load_csv_file(patient_dir + '/ct.csv')
        CT = np.zeros((128, 128, 128), dtype=np.int16)
        indices_ = np.int64(CT_csv['indices'])
        data_ = np.int16(CT_csv['data'])
        np.put(CT, indices_, data_)
        CT = CT - 1024

        # Data in OpenKBP dataset is (h, w, -z) or (y, x, -z)
        CT = CT[:, :, ::-1].transpose([2, 0, 1])
        CT_nii = np2NITFI(CT, spacing)
        sitk.WriteImage(CT_nii, save_patient_path + '/CT.nii.gz')

        # Dose
        dose_csv = load_csv_file(patient_dir + '/dose.csv')
        dose = np.zeros((128, 128, 128), dtype=np.float32)
        indices_ = np.int64(dose_csv['indices'])
        data_ = np.float32(dose_csv['data'])
        np.put(dose, indices_, data_)

        dose = dose[:, :, ::-1].transpose([2, 0, 1])
        dose_nii = np2NITFI(dose, spacing)
        sitk.WriteImage(dose_nii, save_patient_path + '/dose.nii.gz')

        # OARs
        for structure_name in ['PTV70',
                               'PTV63',
                               'PTV56',
                               'possible_dose_mask',
                               'Brainstem',
                               'SpinalCord',
                               'RightParotid',
                               'LeftParotid',
                               'Esophagus',
                               'Larynx',
                               'Mandible']:
            structure_csv_file = patient_dir + '/' + structure_name + '.csv'
            if os.path.exists(structure_csv_file):
                structure_csv = load_csv_file(structure_csv_file)
                structure = np.zeros((128, 128, 128), dtype=np.uint8)
                np.put(structure, structure_csv, np.uint8(1))

                structure = structure[:, :, ::-1].transpose([2, 0, 1])
                structure_nii = np2NITFI(structure, spacing)
                sitk.WriteImage(structure_nii, save_patient_path + '/' + structure_name + '.nii.gz')

        print(patient_id + ' done !')
