import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, ProgressBarBase
from pytorch_lightning.callbacks.progress import RichProgressBar
import os
import sys
sys.path.insert(0, '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/')
from collections import OrderedDict
import json
from monai.config import print_config
import matplotlib.pyplot as plt

import numpy as np

from RTDosePrediction.Src.C3D.model import *
from RTDosePrediction.Src.DataLoader.dataloader_OpenKBP_C3D_monai import get_dataset
import RTDosePrediction.Src.DataLoader.config as config
from RTDosePrediction.Src.Evaluate.evaluate_openKBP import *
from RTDosePrediction.Src.C3D.loss import Loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


# print_config()


class CascadeUNet(pl.LightningModule):
    def __init__(self, pretrain=False):
        super().__init__()
        self.val_data = None
        self.train_data = None

        # OAR + PTV + CT => dose
        self._model = Model(in_ch=9, out_ch=1,
                            list_ch_A=[-1, 16, 32, 64, 128, 256],
                            list_ch_B=[-1, 32, 64, 128, 256, 512],
                            mode_decoder_A=4,
                            mode_decoder_B=4,
                            mode_encoder_A=1,
                            mode_encoder_B=1)

        if pretrain:
            model_dict = self._model.state_dict()
            pretrain_state_dict = torch.load(
                '/pretrained_model/UNETR_model_best_acc.pth')

            pretrain_state_dict = {
                k: v for k, v in pretrain_state_dict.items() if
                (k in model_dict) and (model_dict[k].shape == pretrain_state_dict[k].shape)
            }
            model_dict.update(pretrain_state_dict)
            self._model.load_state_dict(model_dict)

        self.loss_function = Loss()

        self.best_metric = 0
        self.best_metric_epoch = 0
        self.train_epoch_loss = []
        self.metric_values = []

        # self.max_epochs = 1300
        self.max_epochs = 99999
        # 10
        self.check_val = 10
        # 5
        self.warmup_epochs = 5

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch_data, batch_idx):
        input_ = batch_data['Input'].float()
        target = batch_data['GT']

        # train
        output = self.forward(input_)

        train_loss = self.loss_function(output, target)

        tensorboard_logs = {"train_loss": train_loss.item()}

        return {"loss": train_loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.train_epoch_loss.append(train_mean_loss.detach().cpu().numpy())

    def validation_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        sw_batch_size = 4
        prediction_B = sliding_window_inference(
            input_, roi_size, sw_batch_size, lambda x: self.forward(x)[1]
        )
        prediction_B = np.array(prediction_B.cpu())

        # Post processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_B < 0)
        prediction_B[mask] = 0
        Dose_score = 70. * get_3D_Dose_dif(prediction_B.squeeze(0), gt_dose.squeeze(0),
                                           possible_dose_mask.squeeze(0))
        self.list_Dose_score.append(Dose_score)

        # val_loss = self.loss_function(prediction_B, gt_dose)
        # self.log("val_loss", val_loss.item(), prog_bar=True)

        return {"Dose_score": Dose_score, "val_number": len(self.list_Dose_score)}

    def validation_epoch_end(self, outputs):
        # val_loss, num_items = 0, 0
        # for output in outputs:
        #     val_loss += output["val_loss"].sum().item()
        #     num_items += output["val_number"]
        #
        # mean_val_loss = torch.tensor(val_loss / num_items)

        # self.log("mean_val_loss", mean_val_loss, prog_bar=True)

        mean_dose_score = - np.mean(self.list_Dose_score)
        self.log("mean_dose_score", mean_dose_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        tensorboard_logs = {"val_metric": mean_dose_score}

        if mean_dose_score > self.best_metric:
            self.best_metric = mean_dose_score
            self.best_metric_epoch = self.current_epoch

        self.metric_values.append(mean_dose_score)
        return {"log": tensorboard_logs}

    def prepare_data(self):
        self.train_data = get_dataset(path=config.MAIN_PATH + config.TRAIN_DIR, state='train',
                                      size=24, cache=True)

        self.val_data = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='val',
                                    size=6, cache=True)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS, pin_memory=True)
        return val_loader


def main(pretrain=False):
    # initialise the LightningModule
    net = CascadeUNet(pretrain)

    # # set up checkpoints
    # checkpoint_callback = ModelCheckpoint(dirpath=config.CHECKPOINT_MODEL_DIR, filename="best_metric_model")

    # # initialise Lightning's trainer.
    # trainer = pl.Trainer(
    #     devices=[0],
    #     accelerator="gpu",
    #     max_epochs=net.max_epochs,
    #     check_val_every_n_epoch=net.check_val,
    #     callbacks=[checkpoint_callback],
    #     # callbacks=RichProgressBar(),
    #     # callbacks=[bar],
    #     default_root_dir=config.CHECKPOINT_MODEL_DIR,
    #     # enable_progress_bar=True,
    #     # log_every_n_steps=10,
    #     # resume_from_checkpoint="/content/drive/MyDrive/thesis/U-Net/pretrained_model/UNETR_model_best_acc.pth",
    # )

    # # train
    # trainer.fit(net)

    return net


if __name__ == '__main__':
    net = main()
