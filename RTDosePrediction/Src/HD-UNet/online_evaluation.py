# -*- encoding: utf-8 -*-
from DataLoader.dataloader_OpenKBP_C3D import val_transform, read_data, pre_processing
from Evaluate.evaluate_openKBP import *
from model import *


def online_evaluation(trainer):
    list_patient_dirs = ['../../Data/OpenKBP_C3D/pt_' + str(i) for i in range(201, 241)]

    list_Dose_score = []

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in list_patient_dirs:
            patient_name = patient_dir.split('/')[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            gt_dose = list_images[1]
            possible_dose_mask = list_images[2]

            # Forward
            [input_] = val_transform([input_])
            input_ = input_.unsqueeze(0).to(trainer.setting.device)
            [prediction] = trainer.setting.network(input_)
            prediction = np.array(prediction.cpu().data[0, :, :, :, :])

            # Post processing and evaluation
            prediction[np.logical_or(possible_dose_mask < 1, prediction < 0)] = 0
            Dose_score = 70. * get_3D_Dose_dif(prediction.squeeze(0), gt_dose.squeeze(0),
                                               possible_dose_mask.squeeze(0))
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
    # Evaluation score is the higher the better
    return - np.mean(list_Dose_score)
