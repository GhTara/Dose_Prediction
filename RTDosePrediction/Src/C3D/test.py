# -*- encoding: utf-8 -*-
import os
import sys
import argparse
from tqdm import tqdm
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from Evaluate.evaluate_openKBP import *
from model import *
from NetworkTrainer.network_trainer import *


def read_data(patient_dir):
    dict_images = {}
    list_structures = ['CT',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'possible_dose_mask',
                       'Brainstem',
                       'SpinalCord',
                       'RightParotid',
                       'LeftParotid',
                       'Esophagus',
                       'Larynx',
                       'Mandible']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

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
    OAR_all = np.concatenate([dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0)

    # CT image
    CT = dict_images['CT']
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.

    # Possible mask
    possible_dose_mask = dict_images['possible_dose_mask']

    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
                   possible_dose_mask]
    return list_images


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


# Input is C*Z*H*W
def flip_3d(input_, list_axes):
    if 'Z' in list_axes:
        input_ = input_[:, ::-1, :, :]
    if 'W' in list_axes:
        input_ = input_[:, :, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_prediction_B = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        [_, prediction_B] = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction_B = flip_3d(np.array(prediction_B.cpu().data[0, :, :, :, :]), list_flip_axes)

        list_prediction_B.append(prediction_B[0, :, :, :])

    return np.mean(list_prediction_B, axis=0)


def inference(trainer, list_patient_dirs, save_path, do_TTA=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs):
            patient_id = patient_dir.split('/')[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            possible_dose_mask = list_images[1]

            # Test-time augmentation
            if do_TTA:
                TTA_mode = [[], ['Z'], ['W'], ['Z', 'W']]
            else:
                TTA_mode = [[]]
            prediction = test_time_augmentation(trainer, input_, TTA_mode)

            # Pose-processing
            prediction[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)] = 0
            prediction = 70. * prediction

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(patient_dir + '/possible_dose_mask.nii.gz')
            prediction_nii = sitk.GetImageFromArray(prediction)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + patient_id):
                os.mkdir(save_path + '/' + patient_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + patient_id + '/dose.nii.gz')


if __name__ == "__main__":
    if not os.path.exists('../../Data/OpenKBP_C3D'):
        raise Exception('OpenKBP_C3D should be prepared before testing, please run prepare_OpenKBP_C3D.py')

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str, default='../../Output/C3D/best_val_evaluation_index.pkl')
    parser.add_argument('--TTA', type=bool, default=True,
                        help='do test-time augmentation, default True')
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'C3D'
    trainer.setting.output_dir = '../../Output/C3D'

    trainer.setting.network = Model(in_ch=9, out_ch=1,
                                    list_ch_A=[-1, 16, 32, 64, 128, 256],
                                    list_ch_B=[-1, 32, 64, 128, 256, 512])

    # Load model weights
    trainer.init_trainer(ckpt_file=args.model_path,
                         list_GPU_ids=[args.GPU_id],
                         only_network=True)

    # Start inference
    print('\n\n# Start inference !')
    list_patient_dirs = ['../../Data/OpenKBP_C3D/pt_' + str(i) for i in range(241, 341)]
    inference(trainer, list_patient_dirs, save_path=trainer.setting.output_dir + '/Prediction', do_TTA=args.TTA)

    # Evaluation
    print('\n\n# Start evaluation !')
    Dose_score, DVH_score = get_Dose_score_and_DVH_score(prediction_dir=trainer.setting.output_dir + '/Prediction',
                                                         gt_dir='../../Data/OpenKBP_C3D')

    print('\n\nDose score is: ' + str(Dose_score))
    print('DVH score is: ' + str(DVH_score))

