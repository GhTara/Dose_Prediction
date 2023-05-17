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
from RTDosePrediction.Src.C3D.loss import Loss, DiscLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar


class GAN(pl.LightningModule):
    def __init__(
            self,
            n_critic=4,
            pretrain=False,
            lr_scheduler_type='cosine',
            # Adam hp
            lr: float = 3e-4,
            lr_encoder: float = None,
            lr_decoder: float = None,
            weight_decay: float = 1e-4,
            b1_gen: float = 0.5,
            b2_gen: float = 0.999,
            b1_disc: float = 0.5,
            b2_disc: float = 0.999,
            # Cosine hp
            eta_min=None,
            # Step hp
            milestones=None,
            gamma=None,
            last_epoch=None,
            # ReduceLROnPlateau hp
            factor=None,
            patience=None,
            threshold=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_critic = n_critic

        # OAR + PTV + CT => dose
        self.generator = Model(in_ch=9, out_ch=1,
                               list_ch_A=[-1, 16, 32, 64, 128, 256],
                               list_ch_B=[-1, 32, 64, 128, 256, 512])

        # self.discriminator = Discriminator2(in_ch_a=9, in_ch_b=1)
        # self.discriminator = Discriminator3(
        #                         in_channels=10,
        #                         img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
        #                         )
        self.discriminator = Discriminator1_2(in_ch_a=9, in_ch_b=1)

        self.lr_scheduler_type = lr_scheduler_type

        self.val_data = None
        self.train_data = None

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

        self.recon_criterion = Loss()
        # disc1 and disc2
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        # for vit disc
        # self.adversarial_criterion = DiscLoss

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

        self.img_height = config.IMAGE_SIZE
        self.img_width = config.IMAGE_SIZE
        self.img_depth = config.IMAGE_SIZE

        self.list_Dose_score = []

    def forward(self, x):
        return self.generator(x)

    def forward_val(self, x):
        return self.generator(x)[1]

    def training_step(self, batch, batch_idx, optimizer_idx):
        conditioned_image = batch['Input'].float()
        real_image = batch['GT']
        loss = None
        
        

        # Train generator
        if optimizer_idx == 0:
            fake_image = self(conditioned_image)
            disc_logits = self.discriminator(fake_image[1], conditioned_image)

            # Adversarial loss is binary cross-entropy
            adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

            # added for vit
            # adversarial_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - disc_logits))
            
            # Voxel-wise loss
            recon_loss = self.recon_criterion(fake_image, real_image)
            # Total loss
            g_loss = adversarial_loss + config.LAMBDA_VOXEL * recon_loss

            loss = g_loss
            self.log("g_loss", g_loss.item(), prog_bar=True)
        # Train discriminator
        elif optimizer_idx == 1:
            sch = self.lr_schedulers()
            real_logits = self.discriminator(real_image[:, :1, :, :, :], conditioned_image)
            fake_image = self(conditioned_image)
            fake_logits = self.discriminator(fake_image[1].detach(), conditioned_image)

            # for non-vit
            real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
            fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            # add for vit
            # d_loss = self.adversarial_criterion(real_logits, fake_logits)
            
            loss = d_loss

            self.log("d_loss", d_loss.item(), prog_bar=True)
            
            if (self.current_epoch) % self.n_critic == 0:
                sch.step()

        return loss

    # def training_epoch_end(self, outputs):
    #     train_mean_loss = torch.stack([x["g_loss"] for x in outputs]).mean()
    #     self.train_epoch_loss.append(train_mean_loss.detach().cpu().numpy())

    def validation_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        sw_batch_size = 4
        prediction_B = sliding_window_inference(
            input_, roi_size, sw_batch_size, self.forward_val
        )
        prediction_B = np.array(prediction_B.cpu())

        # Post processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_B < 0)
        prediction_B[mask] = 0
        Dose_score = 70. * get_3D_Dose_dif(prediction_B.squeeze(0), gt_dose.squeeze(0),
                                           possible_dose_mask.squeeze(0))
        self.list_Dose_score.append(Dose_score)

        # val_loss = self.loss_function(prediction_B, gt_dose)
        self.log("Dose_score", Dose_score, prog_bar=True)

        return None

    def validation_epoch_end(self, outputs):
        # val_loss, num_items = 0, 0
        # for output in outputs:
        #     val_loss += output["val_loss"].sum().item()
        #     num_items += output["val_number"]

        # mean_val_loss = torch.tensor(val_loss / num_items)

        # self.log("mean_val_loss", mean_val_loss, prog_bar=True)

        mean_dose_score = - np.mean(self.list_Dose_score)
        self.log("mean_dose_score", mean_dose_score, prog_bar=True)

        return {"mean_dose_score": mean_dose_score}

    def configure_optimizers(self):
        lr = self.hparams.lr
        lr_encoder = self.hparams.lr_encoder
        lr_decoder = self.hparams.lr_decoder
        weight_decay = self.hparams.weight_decay
        b1_disc = self.hparams.b1_disc
        b2_disc = self.hparams.b2_disc

        T_max = self.max_epochs
        eta_min = self.hparams.eta_min

        milestones = self.hparams.milestones
        gamma = self.hparams.gamma

        factor = self.hparams.factor
        patience = self.hparams.patience
        threshold = self.hparams.threshold

        last_epoch = self.hparams.last_epoch

        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1_disc, b2_disc))
        # disc_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.discriminator.parameters()),
        # lr=lr, eps=1e-08, weight_decay=weight_decay, momentum=0, centered=False)

        # dis_scheduler = LinearLrDecay(optim_dis, lr, 0.0, 0, args.max_iter * args.n_critic)

        if hasattr(self.generator, 'decoder') and hasattr(self.generator, 'encoder'):
            optimizer = torch.optim.Adam([
                {'params': self.generator.encoder.parameters(), 'lr': lr_encoder},
                {'params': self.generator.decoder.parameters(), 'lr': lr_decoder}
            ],
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True)
        else:
            optimizer = torch.optim.Adam(self.generator.parameters(),
                                             lr=lr,
                                             weight_decay=3e-5,
                                             betas=(0.9, 0.999),
                                             eps=1e-08,
                                             amsgrad=True)

        if self.lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=T_max,
                                                                   eta_min=eta_min,
                                                                   last_epoch=last_epoch
                                                                   )
        elif self.lr_scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=milestones,
                                                             gamma=gamma,
                                                             last_epoch=last_epoch
                                                             )
        elif self.lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=factor,
                                                                   patience=patience,
                                                                   verbose=True,
                                                                   threshold=threshold,
                                                                   threshold_mode='rel',
                                                                   cooldown=0,
                                                                   min_lr=0,
                                                                   eps=1e-08)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss",
                },
            }
        gen_optimizer = optimizer

        return {"optimizer": gen_optimizer, "lr_scheduler": scheduler}, {"optimizer": disc_optimizer}

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


class LitProgressBar(ProgressBarBase):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch_idx)  # don't forget this :)
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')


def main(pretrain=False, n_critic=1):
    # initialise the LightningModule
    net = GAN(
        n_critic = n_critic,
        pretrain=False,
        lr_scheduler_type='cosine',
        lr=3e-4,
        weight_decay=1e-4,
        eta_min=1e-7,
        last_epoch=-1
    )

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(dirpath=config.CHECKPOINT_MODEL_DIR, filename="best_metric_model")
    bar = LitProgressBar()

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        # callbacks=RichProgressBar(),
        # callbacks=[bar],
        default_root_dir=config.CHECKPOINT_MODEL_DIR,
        # enable_progress_bar=True,
        # log_every_n_steps=10,
        # resume_from_checkpoint="/content/drive/MyDrive/thesis/U-Net/pretrained_model/UNETR_model_best_acc.pth",
    )

    # train
    trainer.fit(net)

    return net


if __name__ == '__main__':
    net_ = main()
