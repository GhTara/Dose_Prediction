# -*- encoding: utf-8 -*-
from RTDosePrediction.Src.DataLoader.dataloader_OpenKBP_C3D import val_transform, read_data, pre_processing
from RTDosePrediction.Src.Evaluate.evaluate_openKBP import *
from model import *


def online_evaluation(trainer):

    list_Dose_score = []

    with torch.no_grad():
        trainer.setting.network.eval()
        for batch_idx, list_loader_output in enumerate(trainer.setting.val_loader):
            # List_loader_output[0] default as the input
            input_ = list_loader_output["Input"].float().to(trainer.setting.device)
            target = list_loader_output["GT"]
            gt_dose = np.array(target[:, :1, :, :, :].cpu())
            possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

            # Forward
            [_, prediction_B] = trainer.setting.network(input_)
            prediction_B = np.array(prediction_B.cpu())

            # Post processing and evaluation
            mask = np.logical_or(possible_dose_mask < 1, prediction_B < 0)
            prediction_B[mask] = 0
            Dose_score = 70. * get_3D_Dose_dif(prediction_B.squeeze(0), gt_dose.squeeze(0),
                                               possible_dose_mask.squeeze(0))
            list_Dose_score.append(Dose_score)

            try:
                # how to get name of the patient in monai dataloader
                patient_name = list_loader_output.info()["filename"]
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