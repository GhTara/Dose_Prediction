import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class Loss(nn.Module):
    def __init__(self, casecade=True):
        super().__init__()
        self.casecade = casecade
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, freez=True):
        if self.casecade:
            pred_A = pred[0]
            pred_B = pred[1]
            gt_dose = gt[:, 0:1, :, :, :]
            # gt_dose.squeeze_(1)
            possible_dose_mask = gt[:, 1:, :, :, :]
            # possible_dose_mask.squeeze_(1)
    
            pred_A = pred_A[possible_dose_mask > 0]
            pred_B = pred_B[possible_dose_mask > 0]
            gt_dose = gt_dose[possible_dose_mask > 0]
            
            if freez:
                L1_loss = self.L1_loss_func(pred_B, gt_dose)
            else:
                L1_loss = 0.5 * self.L1_loss_func(pred_A, gt_dose) + self.L1_loss_func(pred_B, gt_dose)
        else:
            gt_dose = gt[:, 0:1, :, :, :]
            # gt_dose.squeeze_(1)
            possible_dose_mask = gt[:, 1:, :, :, :]
            # possible_dose_mask.squeeze_(1)
    
            pred = pred[possible_dose_mask > 0]
            gt_dose = gt_dose[possible_dose_mask > 0]
    
            L1_loss = self.L1_loss_func(pred, gt_dose)
            
        return L1_loss
        
        
def DiscLoss(real_valid, fake_valid):
    loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)) + \
               torch.mean(nn.ReLU(inplace=True)(1 + fake_valid))
    return loss_dis

class GenLoss(nn.Module):
    def __init__(self, im_size=128):
        super().__init__()
        self.perNet = nn.HuberLoss(reduction='mean', delta=0.5)
        self.ds = nn.L1Loss(reduction='mean')
        self.im_size = im_size

    def downSample(self, volume, mask):
        volumes = []
        masks = []
        ch = volume.shape[-1]
        # 4 is depth
        for i in range(1, 4):
            dim = self.im_size // np.power(2, i)
            volume_int = interpolate(volume, size=(dim, dim, dim), mode='trilinear', align_corners=True)
            mask_int = interpolate(mask, size=(dim, dim, dim), mode='nearest-exact')
            volumes.append(volume_int)
            masks.append(mask_int)
        return volumes, masks

    def forward(self, preds, gt, delta1=10, delta2=1, mode='train', casecade=False, freez=True, huber=False):
        gt_dose = gt[:, 0:1, :, :, :]
        possible_dose_mask = gt[:, 1:, :, :, :]
            
        if mode=='train':
            if casecade:
                pred_A = preds[0]
                pred_B = preds[1]
    
                pred_A = pred_A[possible_dose_mask > 0]
        
                preds = pred_B
            # delta1 : [2, 16], delta2: [0.1, 1.7]
            # final out
            pred = preds[0]
            # intermediate out
            pred_intermediate = preds[1:]
    
            gt_intermediate, mask_intermediate = self.downSample(gt_dose, possible_dose_mask)
    
            l_ds = 0
            for i, (pred_i, gt_i) in enumerate(zip(pred_intermediate, gt_intermediate)):
                pred_i = pred_i[mask_intermediate[i] > 0]
                gt_i = gt_i[mask_intermediate[i] > 0]
                l_ds += self.ds(pred_i, gt_i)
            # l_ds += self.ds(pred, gt_dose)
            # l_ds /= len(preds)
            l_ds /= len(pred_intermediate)
    
            pred = pred[possible_dose_mask > 0]
            gt_dose = gt_dose[possible_dose_mask > 0]
    
            if huber:
                l_prenet = self.perNet(pred, gt_dose)
            else:
                l_prenet = self.ds(pred, gt_dose)
            
            loss = delta1 * l_prenet + delta2 * l_ds
            
            if casecade and not freez:
                loss =  loss + 0.5 * self.ds(pred_A, gt_dose)
        else:
            pred = preds[possible_dose_mask > 0]
            gt_dose = gt_dose[possible_dose_mask > 0]
            
            if huber:
                loss = self.perNet(pred, gt_dose) + self.ds(pred, gt_dose)
            else:
                loss = self.ds(pred, gt_dose)

        return loss
