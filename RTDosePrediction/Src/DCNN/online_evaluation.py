# -*- encoding: utf-8 -*-
from DataLoader.dataloader_OpenKBP_DCNN import val_transform, read_data, pre_processing
from Evaluate.evaluate_openKBP import *
from model import *


def online_evaluation(trainer):
    list_patient_dirs = ['../../Data/OpenKBP_DCNN/pt_' + str(i) for i in range(201, 241)]

    list_Dose_score = []

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in list_patient_dirs:
            patient_name = patient_dir.split('/')[-1]

            prediction_dose = np.zeros((128, 128, 128), np.float32)
            gt_dose = np.zeros((128, 128, 128), np.float32)
            possible_dose_mask = np.zeros((128, 128, 128), np.uint8)

            for slice_i in range(128):
                if not os.path.exists(patient_dir + '/CT_' + str(slice_i) + '.nii.gz'):
                    continue

                # Read data and pre-process
                dict_images = read_data(patient_dir, slice_i)
                list_images = pre_processing(dict_images)

                # Forward
                input_ = list_images[0]
                [input_] = val_transform([input_])
                input_ = input_.unsqueeze(0).to(trainer.setting.device)
                [prediction_single_slice] = trainer.setting.network(input_)
                prediction_single_slice = np.array(prediction_single_slice.cpu().data[0, 0, :, :])

                prediction_dose[slice_i, :, :] = prediction_single_slice
                gt_dose[slice_i, :, :] = list_images[1][0, :, :]
                possible_dose_mask[slice_i, :, :] = list_images[2][0, :, :]

            # Post processing and evaluation
            prediction_dose[np.logical_or(possible_dose_mask < 1, prediction_dose < 0)] = 0
            Dose_score = 70. * get_3D_Dose_dif(prediction_dose, gt_dose,
                                               possible_dose_mask)
            list_Dose_score.append(Dose_score)

            try:
                trainer.print_log_to_file('========> ' + patient_name + ':  ' + str(Dose_score), 'a')
            except:
                pass

    try:
        trainer.print_log_to_file('===============================================> mean Dose score: '
                                  + str(np.mean(list_Dose_score)), 'a')
    except:
        pass
    return - np.mean(list_Dose_score)
