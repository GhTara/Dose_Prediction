import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm
import random
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from monai.metrics import DiceMetric

"""
These codes are modified from https://github.com/ababier/open-kbp and https://github.com/LSL000UD/RTDosePrediction 
"""


# define the IVS function
def IVS(pred, gt, isodose_level, possible_dose_mask=None):
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    # # calculate the volumes of the predicted and ground truth isodose regions
    # pred_iso_vol = np.sum(pred >= isodose_level)
    # gt_iso_vol = np.sum(gt >= isodose_level)

    # # calculate the IVS score
    # ivs = 2 * (pred_iso_vol * gt_iso_vol) / ((pred_iso_vol + gt_iso_vol) ** 2)

    # calculate the volumes of the predicted and ground truth isodose regions
    pred_iso_vol = pred >= isodose_level
    gt_iso_vol = gt >= isodose_level

    # calculate the IVS score
    ivs = 2 * np.sum(pred_iso_vol * gt_iso_vol) / (np.sum(pred_iso_vol) + np.sum(gt_iso_vol))

    return ivs


def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    dif = np.mean(np.abs(pred - gt))
    return dif


def get_DVH_metrics(_dose, _mask, mode, spacing=None):
    output = {}

    if mode == 'target':
        _roi_dose = _dose[_mask > 0]
        # D1
        output['D1'] = np.percentile(_roi_dose, 99)
        # D95
        output['D95'] = np.percentile(_roi_dose, 5)
        # D99
        output['D99'] = np.percentile(_roi_dose, 1)
        # Dmean
        output['mean'] = np.mean(_roi_dose)

    elif mode == 'OAR':
        if spacing is None:
            raise Exception('calculate OAR metrics need spacing')

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        _voxel_size = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        # D_0.1_cc
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        output['D_0.1_cc'] = np.percentile(_roi_dose, fractional_volume_to_evaluate)
        # Dmean
        output['mean'] = np.mean(_roi_dose)
    else:
        raise Exception('Unknown mode!')

    return output


def get_Dose_score_and_DVH_score(prediction_dir, gt_dir):
    list_dose_dif = []
    list_DVH_dif = []
    gt_list_DVH = {}
    pred_list_DVH = {}
    metric_dif = {}

    list_patient_ids = tqdm(os.listdir(prediction_dir))
    for patient_id in list_patient_ids:
        pred_nii = sitk.ReadImage(prediction_dir + '/' + patient_id + '/dose.nii.gz')
        pred = sitk.GetArrayFromImage(pred_nii)

        gt_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/dose.nii.gz')
        gt = sitk.GetArrayFromImage(gt_nii)

        # Dose dif
        possible_dose_mask_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/possible_dose_mask.nii.gz')
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
        list_dose_dif.append(get_3D_Dose_dif(pred, gt, possible_dose_mask))

        # DVH dif
        for structure_name in ['Brainstem',
                               'SpinalCord',
                               'RightParotid',
                               'LeftParotid',
                               'Esophagus',
                               'Larynx',
                               'Mandible',

                               'PTV70',
                               'PTV63',
                               'PTV56']:
            structure_file = gt_dir + '/' + patient_id + '/' + structure_name + '.nii.gz'

            # If the structure has been delineated
            if os.path.exists(structure_file):
                structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
                structure = sitk.GetArrayFromImage(structure_nii)

                spacing = structure_nii.GetSpacing()
                if structure_name.find('PTV') > -1:
                    mode = 'target'
                else:
                    mode = 'OAR'
                pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
                gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

                for metric in gt_DVH.keys():
                    list_DVH_dif.append(abs(gt_DVH[metric] - pred_DVH[metric]))
                    if not metric_dif[metric]:
                        metric_dif[metric] = []
                        gt_list_DVH[metric] = []
                        pred_list_DVH[metric] = []
                    metric_dif[metric].append(abs((gt_DVH[metric] - pred_DVH[metric])))
                    gt_list_DVH[metric].append(gt_DVH[metric])
                    pred_list_DVH[metric].append(pred_DVH[metric])

    for key in gt_list_DVH.keys():
        gt_list_DVH[key] = np.mean(gt_list_DVH[key])
        pred_list_DVH[key] = np.mean(pred_list_DVH[key])
        metric_dif[key] = np.mean(metric_dif[key])

    return np.mean(list_dose_dif), np.mean(list_DVH_dif), gt_list_DVH, pred_list_DVH, metric_dif


def get_Dose_score_and_DVH_score_batch(prediction, batch_data, list_DVH_dif, dict_DVH_dif={}, ivs_values=[]):
    list_DVH_dif = []

    # pred = np.array(prediction[0, 0, :, :, :])
    pred = np.array(prediction[0, 0, :, :, :].cpu())
    gt = np.array(batch_data['real_dose'][0, 0, :, :, :].cpu())

    # Dose dif
    possible_dose_mask = np.array(batch_data['dose_mask'][0, 0, :, :, :].cpu())
    dose_dif = get_3D_Dose_dif(pred, gt, possible_dose_mask)

    # IVS
    # define the isodose levels to calculate IVS at
    # if True:
    if ivs_values is not None:
        isodose_levels = np.linspace(0, 70, 101)
        ivs_values.append([])
        for isoLevel in isodose_levels:
            ivs = IVS(pred, gt, isoLevel, possible_dose_mask=None)
            ivs_values[-1].append(ivs)

    structure_file = batch_data['file_path'][0]
    patient_id = structure_file.split('/')[-2]
    dict_DVH_dif[patient_id] = {}
    keys = batch_data.keys()

    # DVH dif
    for structure_name in ['Brainstem',
                           'SpinalCord',
                           'RightParotid',
                           'LeftParotid',
                           'Esophagus',
                           'Larynx',
                           'Mandible',

                           'PTV70',
                           'PTV63',
                           'PTV56']:

        if not (structure_name in keys):
            break

        # if not (structure_name in dict_DVH_dif.keys()):
        #     dict_DVH_dif[patient_id] = {}

        structure = np.array(batch_data[structure_name][0, 0, :, :, :].cpu())
        # If the structure has been delineated
        if np.any(structure):
            structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            spacing = structure_nii.GetSpacing()
            if structure_name.find('PTV') > -1:
                mode = 'target'
            else:
                mode = 'OAR'
                # structure = structure[0]
            pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
            gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

            for metric in gt_DVH.keys():
                list_DVH_dif.append(abs(gt_DVH[metric] - pred_DVH[metric]))

                # if not (metric in dict_DVH_dif[structure_name].keys()):
                #     dict_DVH_dif[structure_name][metric] = []
                col_name = 'pre' + structure_name + '_' + metric
                # dict_DVH_dif[patient_id][col_name] = abs(gt_DVH[metric] - pred_DVH[metric])
                dict_DVH_dif[patient_id][col_name] = pred_DVH[metric]

                col_name = 'gt_' + structure_name + '_' + metric
                dict_DVH_dif[patient_id][col_name] = gt_DVH[metric]

    dict_DVH_dif[patient_id]['_dose_dif'] = dose_dif
    dict_DVH_dif[patient_id]['_DVH_dif'] = np.mean(list_DVH_dif)

    return dose_dif, np.mean(list_DVH_dif), dict_DVH_dif, ivs_values


def plot_DVH(prediction, batch_data, path):
    new_dose = np.array(prediction[0, 0, :, :, :].cpu()).flatten()
    reference_dose = np.array(batch_data['real_dose'][0, 0, :, :, :].cpu()).flatten()
    structure_file = batch_data['file_path'][0]
    keys = batch_data.keys()

    structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
    spacing = structure_nii.GetSpacing()

    _voxel_size = np.prod(spacing)
    voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))

    DVH_bin = 5000
    DVH_inv = 90.0 * 1.0 / DVH_bin
    dose_bin = np.zeros(DVH_bin)
    dose_bin_plot = np.arange(0, DVH_bin) * DVH_inv
    dose_bin = np.arange(-1, DVH_bin) * DVH_inv
    DVH_all_ref = defaultdict()
    DVH_all_pred = defaultdict()
    colors = mcolors.TABLEAU_COLORS
    colors_labels = {'Brainstem': colors[list(colors.keys())[0]],
                     'SpinalCord': colors[list(colors.keys())[1]],
                     'RightParotid': colors[list(colors.keys())[2]],
                     'LeftParotid': colors[list(colors.keys())[3]],
                     'Esophagus': colors[list(colors.keys())[4]],
                     'Larynx': colors[list(colors.keys())[5]],
                     'Mandible': colors[list(colors.keys())[6]],

                     'PTV70': colors[list(colors.keys())[7]],
                     'PTV63': colors[list(colors.keys())[8]],
                     'PTV56': colors[list(colors.keys())[9]]}
    for structure_name in ['Brainstem',
                           'SpinalCord',
                           'RightParotid',
                           'LeftParotid',
                           'Esophagus',
                           'Larynx',
                           'Mandible',

                           'PTV70',
                           'PTV63',
                           'PTV56']:

        if not (structure_name in keys):
            break

        roi_mask = np.array(batch_data[structure_name][0].cpu())
        # If the structure has been delineated
        if np.any(roi_mask):
            if structure_name.find('PTV') > -1:
                mode = 'target'
            else:
                mode = 'OAR'
                # roi_mask = roi_mask[0]
            roi_mask = roi_mask.flatten()
            roi_dose_ref = reference_dose[roi_mask > 0]
            roi_dose_pred = new_dose[roi_mask > 0]
            max_dose_ref = np.max(roi_dose_ref)
            max_dose_pred = np.max(roi_dose_pred)
            roi_size = len(roi_dose_ref)
            print('roi size', roi_size)
            DVH = np.zeros(DVH_bin)
            DVH_diff_ref, bin_edges = np.histogram(roi_dose_ref, dose_bin)
            DVH = np.cumsum(DVH_diff_ref)
            DVH = 1 - DVH / DVH.max()
            DVH_all_ref[structure_name] = DVH

            DVH = np.zeros(DVH_bin)
            DVH_diff_ref, bin_edges = np.histogram(roi_dose_pred, dose_bin)
            DVH = np.cumsum(DVH_diff_ref)
            DVH = 1 - DVH / DVH.max()
            DVH_all_pred[structure_name] = DVH

    #   self.calculate_metrics(self.new_dose_metric_df, new_dose)
    fig = plt.figure(dpi=1200)
    roi_legend = []
    for roi in DVH_all_ref.keys():
        r = random.uniform(0, 1);
        g = random.uniform(0, 1);
        b = random.uniform(0, 1)
        line, = plt.plot(dose_bin_plot, DVH_all_ref[roi] * 100, color=colors_labels[roi], linewidth=2, label=roi)
        plt.plot(dose_bin_plot, DVH_all_pred[roi] * 100, color=colors_labels[roi], linewidth=2, linestyle='dashed',
                 label=roi)
        roi_legend.append(line)

    plt.ylabel('volume %')
    plt.legend(handles=roi_legend, bbox_to_anchor=(1.1, 1.05), prop={'size': 6})
    plt.savefig(path, dpi=300)
    plt.show()
    print('dose shape', reference_dose.shape)
