import os
import sys
sys.path.insert(0, '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/')

from statistics import mean 

import SimpleITK as sitk

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate, decollate_batch, NiftiSaver, write_nifti

from torchvision.utils import make_grid
from torchvision.utils import save_image

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, ProgressBarBase
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from ray_lightning import RayShardedStrategy

import bitsandbytes as bnb

from typing import Optional

from RTDosePrediction.Src.C3D.my_model import *
# from RTDosePrediction.Src.C3D.model import *
from RTDosePrediction.Src.DataLoader.dataloader_OpenKBP_C3D_monai import get_dataset
import RTDosePrediction.Src.DataLoader.config as config
from RTDosePrediction.Src.Evaluate.evaluate_openKBP import *
from RTDosePrediction.Src.C3D.loss import Loss, GenLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar


class OpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_data = get_dataset(path=config.MAIN_PATH + config.TRAIN_DIR, state='train',
                                      size=200, cache=True, crop_flag=True)

        self.val_data = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='val',
                                    size=100, cache=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)


class TestOpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # Assign val datasets for use in dataloaders

        self.test_data = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='test',
                                    size=100, cache=True)
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)


class CascadeUNet(pl.LightningModule):
    def __init__(
            self,
            config_param,
            sw_batch_size=1,
            lr_scheduler_type='cosine',
            # Adam hp
            lr: float = 3e-4,
            lr_encoder: float = None,
            lr_decoder: float = None,
            weight_decay: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
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
        self.config_param = config_param
        self.save_hyperparameters()

        # OAR + PTV + CT => dose

        # self.model_ = SharedUNetRModelA(
        # in_channels_a=9,
        # in_channels_b=9,
        # out_channels=1,
        # img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
        # # 16 => 4
        # feature_size=16,
        # hidden_size=768,
        # mlp_dim=3072,
        # num_heads=12,
        # pos_embed="perceptron",
        # norm_name="instance",
        # res_block=True,
        # conv_block=True,
        # dropout_rate=0.0)

        # self.model_ = SharedUNetRModel(
        # in_channels_a=9,
        # in_channels_b=25,
        # out_channels=1,
        # img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
        # # 16 => 4
        # feature_size=16,
        # hidden_size=768,
        # mlp_dim=3072,
        # num_heads=12,
        # pos_embed="perceptron",
        # norm_name="instance",
        # res_block=True,
        # conv_block=True,
        # dropout_rate=0.0)

        # self.model_ = SharedUNetModel(in_ch=9, out_ch=1,
        #                         list_ch=[-1, 16, 32, 64, 128, 256, 512],
        #                         mode_decoder=2,
        #                         mode_encoder=2)
        
        self.model_ = VitGenerator(
            in_ch=9,
            out_ch=1,
            mode_decoder=1,
            mode_encoder=1,
            feature_size=16,
            img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
            num_layers=8,  # 4, 8, 12
            num_heads=6,  # 3, 6, 12,
            mode_multi_dec=True,
        )
        
        # self.model_ = SharedEncoderModel(in_ch=9,
        #                                  out_ch=1,
        #                                  mode_decoder=1,
        #                                  mode_encoder=1,
        #                                  img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
        #                                  num_layers=config_param["num_layers"],  # 4, 8, 12
        #                                  num_heads=config_param["num_heads"]  # 3, 6, 12
        #                                  )

        # self.model_ = Model(in_ch=9, out_ch=1,
        #               list_ch_A=[-1, 16, 32, 64, 128, 256],
        #               list_ch_B=[-1, 32, 64, 128, 256, 512],
        #               mode_decoder_A=4,
        #               mode_decoder_B=4,
        #               mode_encoder_A=2,
        #               mode_encoder_B=1)
        # self.model_ = ModelMonai(in_ch=9, out_ch=1,
        #                   list_ch_A=(16, 32, 64, 128, 256),
        #                   list_ch_B=(32, 64, 128, 256, 512),
        #                   mode_decoder_A=1,
        #                   mode_decoder_B=1,
        #                   mode_encoder_A=1,
        #                   mode_encoder_B=2)

        self.lr_scheduler_type = lr_scheduler_type

        self.loss_function = GenLoss()
        # Moving average loss, loss is the smaller the better
        self.eps_train_loss = 0.01

        self.best_average_val_index = -99999999.
        self.average_val_index = None
        self.metric_values = []

        self.best_average_train_loss = 99999999.
        self.moving_train_loss = None
        self.train_epoch_loss = []

        self.max_epochs = 1300
        # self.max_epochs = 150
        # 10
        self.check_val = 3
        # 5
        self.warmup_epochs = 1

        self.img_height = config.IMAGE_SIZE
        self.img_width = config.IMAGE_SIZE
        self.img_depth = config.IMAGE_SIZE
        
        self.sw_batch_size = sw_batch_size
        
        self.list_DVH_dif = []
        
        self.saver = NiftiSaver(config.CHECKPOINT_RESULT_DIR)

    def forward(self, x):
        return self.model_(x)

    def training_step(self, batch, batch_idx):
        input_ = batch['Input'].float()
        target = batch['GT']

        # train
        output = self(input_)
        torch.cuda.empty_cache()

        loss = self.loss_function(output, target)

        if self.moving_train_loss is None:
            self.moving_train_loss = loss.item()
        else:
            self.moving_train_loss = \
                (1 - self.eps_train_loss) * self.moving_train_loss \
                + self.eps_train_loss * loss.item()

        # self.log("moving_train_loss", self.moving_train_loss, logger=False)

        tensorboard_logs = {"train_loss": loss.item()}

        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if train_mean_loss < self.best_average_train_loss:
            self.best_average_train_loss = train_mean_loss
        self.train_epoch_loss.append(train_mean_loss.detach().cpu().numpy())
        # self.log("best_average_train_loss", self.best_average_train_loss, logger=False)
        self.logger.log_metrics({"train_mean_loss": train_mean_loss}, self.current_epoch + 1)

    def validation_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)

        # prediction = self.forward(input_)
        torch.cuda.empty_cache()
        
        prediction = sliding_window_inference(
            input_, roi_size, self.sw_batch_size, lambda x: self.forward(x)[0]
        )
        loss = self.loss_function(prediction, target, mode='val')

        # prediction_b = np.array(prediction[1].cpu())
        # for vit generator
        prediction_b = np.array(prediction.cpu())

        # Post-processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_b < 0)
        prediction_b[mask] = 0
        dose_score = 80. * get_3D_Dose_dif(prediction_b.squeeze(0), gt_dose.squeeze(0),
                                           possible_dose_mask.squeeze(0))

        return {"val_loss": loss, "val_metric": dose_score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_dose_score = - np.stack([x["val_metric"] for x in outputs]).mean()
        if mean_dose_score > self.best_average_val_index:
            self.best_average_val_index = mean_dose_score
        self.metric_values.append(mean_dose_score)

        self.log("mean_dose_score", mean_dose_score, logger=False)
        self.log("val_loss", avg_loss, logger=False)
        self.logger.log_metrics({"mean_dose_score": mean_dose_score}, self.current_epoch + 1)
        self.logger.log_metrics({"val_loss": avg_loss}, self.current_epoch + 1)

        tensorboard_logs = {"val_metric": mean_dose_score}

        return {"log": tensorboard_logs}

    # def configure_optimizers(self):
    #     lr = self.hparams.lr
    #     lr_encoder = self.hparams.lr_encoder
    #     lr_decoder = self.hparams.lr_decoder
    #     weight_decay = self.hparams.weight_decay

    #     T_max = self.max_epochs
    #     eta_min = self.hparams.eta_min

    #     milestones = self.hparams.milestones
    #     gamma = self.hparams.gamma

    #     factor = self.hparams.factor
    #     patience = self.hparams.patience
    #     threshold = self.hparams.threshold

    #     last_epoch = self.hparams.last_epoch

    #     if hasattr(self.model_, 'decoder') and hasattr(self.model_, 'encoder'):
    #         optimizer = torch.optim.Adam([
    #             {'params': self.model_.encoder.parameters(), 'lr': lr_encoder},
    #             {'params': self.model_.decoder.parameters(), 'lr': lr_decoder}
    #         ],
    #             weight_decay=weight_decay,
    #             betas=(0.9, 0.999),
    #             eps=1e-08,
    #             amsgrad=True)
    #     else:
    #         optimizer = torch.optim.Adam(self.model_.parameters(),
    #                                      lr=lr,
    #                                      weight_decay=3e-5,
    #                                      betas=(0.9, 0.999),
    #                                      eps=1e-08,
    #                                      amsgrad=True)

    #     if self.lr_scheduler_type == 'cosine':
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                               T_max=T_max,
    #                                                               eta_min=eta_min,
    #                                                               last_epoch=last_epoch
    #                                                               )
    #     elif self.lr_scheduler_type == 'step':
    #         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                          milestones=milestones,
    #                                                          gamma=gamma,
    #                                                          last_epoch=last_epoch
    #                                                          )
    #     elif self.lr_scheduler_type == 'ReduceLROnPlateau':
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                               mode='min',
    #                                                               factor=factor,
    #                                                               patience=patience,
    #                                                               verbose=True,
    #                                                               threshold=threshold,
    #                                                               threshold_mode='rel',
    #                                                               cooldown=0,
    #                                                               min_lr=0,
    #                                                               eps=1e-08)
    #         return [optimizer], [{"scheduler": scheduler,
    #                               "monitor": "moving_train_loss",
    #                               }]

    #     return [optimizer], [{"scheduler": scheduler}]

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.model_.parameters(), lr=1e-4, weight_decay=1e-5
        # )
        optimizer = bnb.optim.Adam8bit(self.model_.parameters(), lr=1e-4, weight_decay=1e-5) # instead of torch.optim.Adam
        self.logger.log_hyperparams(params=dict(num_layers=self.config_param["num_layers"],  # 4, 8, 12
                                                num_heads=self.config_param["num_heads"]))
                                                
        # T_max = self.max_epochs
        # eta_min = self.hparams.eta_min
        # last_epoch = self.hparams.last_epoch
                                                
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                           T_max=T_max,
        #                                                           eta_min=eta_min,
        #                                                           last_epoch=last_epoch
        #                                                           )
    
        # return [optimizer], [{"scheduler": scheduler}]
        return optimizer
        
        
    def test_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        
        gt_dose = target[:, :1, :, :, :].cpu()
        possible_dose_mask = target[:, 1:, :, :, :].cpu()

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        # prediction = self.forward(input_)
        prediction = sliding_window_inference(
            input_, roi_size, self.sw_batch_size, lambda x: self.forward(x)[0]
        )
        
        dose_dif, self.list_DVH_dif = get_Dose_score_and_DVH_score_batch(prediction, batch_data, self.list_DVH_dif)
        # plot_DVH(prediction, batch_data, path=os.path.join(config.CHECKPOINT_RESULT_DIR, 'dvh_{}.png'.format(batch_idx)))
        
        torch.cuda.empty_cache()
        
        # prediction_b = prediction.cpu()

        # # Post-processing and evaluation
        # mask = torch.logical_or(possible_dose_mask < 1, prediction_b < 0)
        # prediction_b[mask] = 0
        # gt_dose[mask] = 0
                
        # pred_img = torch.permute(prediction_b[0], (3,0,1,2))
        # gt_img = torch.permute(gt_dose[0], (3,0,1,2))
        
        # # save as image
        # pred_grid = make_grid(pred_img, nrow=11)
        # gt_grid = make_grid(gt_img, nrow=11)
        
        # if not os.path.isdir(config.CHECKPOINT_RESULT_DIR):
        #     os.mkdir(config.CHECKPOINT_RESULT_DIR)
            
        # save_image(pred_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'pred_{}.jpg'.format(batch_idx)))
        # save_image(gt_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'gt_{}.jpg'.format(batch_idx)))
        
        return {"dose_dif": dose_dif}


    def test_epoch_end(self, outputs):
        
        mean_dose_metric = np.stack([x["dose_dif"] for x in outputs]).mean()
        # mean_dvh_metric = torch.stack(self.list_DVH_dif).mean()
        mean_dvh_metric = mean(self.list_DVH_dif)
        
        print(mean_dose_metric, mean_dvh_metric)

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("mean_dvh_metric", mean_dvh_metric)
        # self.logger.log_metrics({"mean_dose_score": mean_dose_score}, self.current_epoch + 1)
        # self.logger.log_metrics({"val_loss": avg_loss}, self.current_epoch + 1)

    
    
def main():
    # initialise the LightningModule
    openkbp = OpenKBPDataModule()
    config_param = {
        "num_layers": 4,
        "num_heads": 6,
        # "lr": tune.loguniform(1e-4, 1e-1),
    }
    net = CascadeUNet(
        config_param,
        lr_scheduler_type='cosine',
        lr=3e-4,
        weight_decay=1e-4,
        eta_min=1e-7,
        last_epoch=-1
    )

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_MODEL_DIR,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
        #  filename=net.filename,
    )

    # set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='/Users/gheshlaghitara@gmail.com/dose_prediction_vitGen',
        tracking_uri="databricks",
        # run_id = '3e56633fa9184531b91d1d632dd3ae99'
        # run_name = 'vitgen_multiS_dec_random_crop_300'
        run_id = 'c31b24d074424ae18fdb0855431b0df0'
        # with scheduler
        # run_name = 'vitgen_multiS_dec_random_crop_300_cosine',
        # run_id = '8cd4f1d9b24749ba950b2f0d7331628a'
        # run_name = 'vitgen_multiS_dec_random_crop_300-no-huber-loss',
        # run_id = ''
    )
    
    strategy = RayShardedStrategy(num_workers=1, num_cpus_per_worker=1, use_gpu=True)

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        # strategy=strategy,
        devices=[0],
        accelerator="gpu",
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        # callbacks=RichProgressBar(),
        # callbacks=[bar],
        default_root_dir=config.CHECKPOINT_MODEL_DIR,
        # enable_progress_bar=True,
        # log_every_n_steps=net.check_val,
    )

    # train
    # trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/results_thesis/re_swin_unetr/last.ckpt')
    trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/other_re_th/last.ckpt')
    # trainer.fit(net, datamodule=openkbp)

    return net


if __name__ == '__main__':
    net_ = main()