# -*- encoding: utf-8 -*-
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt):
        pred_dose = pred[0]
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        pred_dose = pred_dose[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        L1_loss = self.L1_loss_func(pred_dose, gt_dose)
        return L1_loss
