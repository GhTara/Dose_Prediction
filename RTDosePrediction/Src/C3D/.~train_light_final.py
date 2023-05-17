import os
import sys
sys.path.insert(0, '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/')

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate, decollate_batch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, ProgressBarBase
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from ray_lightning import RayShardedStrategy

from torchvision.utils import make_grid
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

import bitsandbytes as bnb

from typing import Optional

import matplotlib.pyplot as plt

from RTDosePrediction.Src.C3D.my_model import *
# from RTDosePrediction.Src.C3D.model import *
from RTDosePrediction.Src.DataLoader.dataloader_OpenKBP_C3D_monai import get_dataset
import RTDosePrediction.Src.DataLoader.config as config
from RTDosePrediction.Src.Evaluate.evaluate_openKBP import *
from RTDosePrediction.Src.C3D.loss import Loss, GenLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


class OpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_data = get_dataset(path=config.MAIN_PATH + config.TRAIN_DIR, state='train',
                                      size=200, cache=True, crop_flag=False)

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
                                    size=3, cache=True)
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)


class CascadeUNet(pl.LightningModule):
    def __init__(
            self,
            config_param,
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
            freez=True
    ):
        super().__init__()
        self.config_param = config_param
        self.freez = freez
        self.save_hyperparameters()

        # OAR + PTV + CT => dose
        
        self.model_, inside = create_pretrained_unet(
        in_ch=9, out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        list_ch_B=[-1, 32, 64, 128, 256, 512],
        ckpt_file='/content/drive/MyDrive/thesis/pretrained_models/baseline/C3D_bs4_iter80000.pkl',
        mode_decoder=1,
        mode_encoder=1,
        feature_size=16,
        img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
        num_layers=8,  # 4, 8, 12
        num_heads=6,  # 3, 6, 12
        # act='mish',
        act=config_param["act"],
        mode_multi_dec=True,
        # multiS_conv=True,
        multiS_conv=config_param["multiS_conv"],)
        
        if freez:
            for n, param in self.model_.named_parameters():
                if  'net_A' in n or 'conv_out_A' in n:
                    param.requires_grad = False
        

        # self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_type = config_param["lr"]
        self.weight_decay = config_param["weight_decay"]

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
        self.check_val = 5
        # 5
        self.warmup_epochs = 1

        self.img_height = config.IMAGE_SIZE
        self.img_width = config.IMAGE_SIZE
        self.img_depth = config.IMAGE_SIZE
        
        self.sw_batch_size = config.SW_BATCH_SIZE
        
        self.list_DVH_dif = []
        self.dict_DVH_dif = {}

    def forward(self, x):
        return self.model_(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch['Input'].float()
        target = batch['GT']

        # train
        output = self(input_)
        torch.cuda.empty_cache()

        loss = self.loss_function(output, target, casecade=True, freez=self.freez,
                                    delta1=self.config_param['delta1'], delta2=self.config_param['delta2'])

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
        torch.cuda.empty_cache()
        train_mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if train_mean_loss < self.best_average_train_loss:
            self.best_average_train_loss = train_mean_loss
        self.train_epoch_loss.append(train_mean_loss.detach().cpu().numpy())
        # self.log("best_average_train_loss", self.best_average_train_loss, logger=False)
        self.logger.log_metrics({"train_mean_loss": train_mean_loss}, self.current_epoch + 1)
        torch.cuda.empty_cache()

    def validation_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
        prediction = self.forward(input_)
        
        
        # prediction = sliding_window_inference(
        #     input_, roi_size, self.sw_batch_size, lambda x: self.forward(x)[1]
        # )
        torch.cuda.empty_cache()
        loss = self.loss_function(prediction[1][0], target, mode='val', casecade=True, freez=self.freez)

        prediction_b = np.array(prediction[1][0].cpu())

        # Post-processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_b < 0)
        prediction_b[mask] = 0
        dose_score = 70. * get_3D_Dose_dif(prediction_b.squeeze(0), gt_dose.squeeze(0),
                                           possible_dose_mask.squeeze(0))

        return {"val_loss": loss, "val_metric": dose_score}

    def validation_epoch_end(self, outputs):
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()

        return {"log": tensorboard_logs}

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.model_.parameters(), lr=1e-3, weight_decay=1e-4)
        # optimizer = bnb.optim.Adam8bit(self.model_.parameters(), lr=1e-3, weight_decay=1e-4) # instead of torch.optim.Adam
        optimizer = bnb.optim.Adam8bit(self.model_.parameters(), lr=self.lr_scheduler_type, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        # return [optimizer], [scheduler]
        return optimizer
        
    def test_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        
        gt_dose = target[:, :1, :, :, :].cpu()
        possible_dose_mask = target[:, 1:, :, :, :].cpu()

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        # sw_batch_size = 1
        prediction = self.forward(input_)
        prediction = prediction[1][0].cpu()
    
        prediction[np.logical_or(possible_dose_mask < 1, prediction < 0)] = 0
        
        # for save image
        # pred_img = torch.permute(prediction[0].cpu(), (1,0,2,3))
        
        prediction = 70. * prediction
        
        dose_dif, self.list_DVH_dif, self.dict_DVH_dif = get_Dose_score_and_DVH_score_batch(
            prediction, batch_data, list_DVH_dif=self.list_DVH_dif, dict_DVH_dif=self.dict_DVH_dif)
        
        torch.cuda.empty_cache()
        if batch_idx<3:
            plot_DVH(prediction, batch_data, path=os.path.join(config.CHECKPOINT_RESULT_DIR, 'dvh_{}.png'.format(batch_idx)))
        
            prediction_b = prediction.cpu()
            

            # Post-processing and evaluation
            # prediction[torch.logical_or(possible_dose_mask < 1, prediction < 0)] = 0
            # for save image
            gt_dose[possible_dose_mask < 1] = 0
            
            pred_img = torch.permute(prediction[0].cpu(), (1,0,2,3))
            gt_img = torch.permute(gt_dose[0], (1,0,2,3))
            
            for i in range(len(pred_img)):
                pred_i = pred_img[i][0].numpy()
                gt_i = gt_img[i][0].numpy()
                plt.figure("check", (12, 6))
                plt.subplots_adjust(wspace=0, hspace=0)
            
                plt.subplot(1, 2, 1)
                plt.title(f"GT")
                plt.imshow(gt_i)
                plt.axis('off')
            
                plt.subplot(1,2, 2)
                plt.title(f"pred")
                plt.imshow(pred_i)
                plt.axis('off')
            
                name_p = batch_data['file_path'].cpu().split("/")[-2]
                print(name_p)
                save_dir = os.path.join(config.CHECKPOINT_RESULT_DIR, '{}_{}'.format(name_p, batch_idx))
                if not os.path.isdir(save_dir):
                            os.mkdir(save_dir)
            
                plt.savefig(os.path.join(save_dir, '{}.jpg'.format(i)), bbox_inches="tight")
            
            # save as image
            # pred_grid = make_grid(pred_img, nrow=11)
            # gt_grid = make_grid(gt_img, nrow=11)
            
            # if not os.path.isdir(config.CHECKPOINT_RESULT_DIR):
            #     os.mkdir(config.CHECKPOINT_RESULT_DIR)
                
            # plt.imsave(os.path.join(config.CHECKPOINT_RESULT_DIR, 'gt_{}.jpg'.format(batch_idx)), gt_grid[0, :, :])
            # plt.imsave(os.path.join(config.CHECKPOINT_RESULT_DIR, 'pred_{}.jpg'.format(batch_idx)), pred_grid[0, :, :])
            # save_image(pred_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'pred_{}.jpg'.format(batch_idx)))
            # save_image(gt_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'gt_{}.jpg'.format(batch_idx)))
        
        return {"dose_dif": dose_dif}


    def test_epoch_end(self, outputs):
        
        mean_dose_metric = np.stack([x["dose_dif"] for x in outputs]).mean()
        # mean_dvh_metric = torch.stack(self.list_DVH_dif).mean()
        mean_dvh_metric = np.mean(self.list_DVH_dif)
        # for key in self.dict_DVH_dif.keys():
        #     for metric in self.dict_DVH_dif[key]:
        #         self.dict_DVH_dif[key][metric] = np.mean(self.dict_DVH_dif[key][metric])
        
        print(mean_dose_metric, mean_dvh_metric)
        print('----------------------Difference DVH for each structures---------------------')
        print(self.dict_DVH_dif)
        # print('----------------------predicted---------------------')
        # print(self.pred_list_DVH)

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("mean_dvh_metric", mean_dvh_metric)
        # self.logger.log_metrics({"mean_dose_score": mean_dose_score}, self.current_epoch + 1)
        # self.logger.log_metrics({"val_loss": avg_loss}, self.current_epoch + 1)
        return self.dict_DVH_dif


def main(freez=True):
    # initialise the LightningModule
    openkbp = OpenKBPDataModule()
    config_param = {
        "act": 'mish',
        "multiS_conv": True,
        # 'lr': 3*1e-4, 
        "lr": 0.0006130697604327541,
        # 'weight_decay': 2*1e-4,
        'weight_decay': 0.00016303111017674179,
        'delta1': 10,
        'delta2': 8,
    }
    net = CascadeUNet(
        config_param,
        lr_scheduler_type='cosine',
        eta_min=1e-7,
        last_epoch=-1,
        freez=freez
    )

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_MODEL_DIR_FINAL_RAY,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
        #  filename=net.filename,
    )

    # set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='/Users/gheshlaghitara@gmail.com/dose_prediction_vitGen',
        tracking_uri="databricks",
        # run_name = 'vitgen_multiS_dec_random_crop_300_p'
        # run_id = '3ac01f8630f345a2b38d813f67ad9a8e'
        # run_name = 'vitgen_multiS_dec_random_crop_300_p2'
        # run_name = 'vitgen_multiS_dec_random_crop_300_p3_32f'
        # run_id = '63d2bc7b84754e17bae840fa6a0c01e3'
        # run_name = 'vitgen_multiS_dec_random_crop_300_p_raytune'
        # run_id = 'b8a4d25100f346e6a8cc93cd91a3b0f6'
        # run_name = 'vitgen_multiS_dec_random_crop_300_p_raytune_ex2'
        # run_id = '9e2351b9cd194dffb805a34e5ed5a3f8'
        # run_name = 'vitgen_multiS_dec_random_crop_300_p_raytune_ex3_batch2'
        run_id = 'de9478bb77c54c6599a23cea061e7ef9'
    )
    
    # {'act': 'mish', 'multiS_conv': True, 'lr': 0.0002840195762381102, 'weight_decay': 0.00021139244378558662
    
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
        default_root_dir=config.CHECKPOINT_MODEL_DIR_FINAL_RAY,
        # enable_progress_bar=True,
        # log_every_n_steps=net.check_val,
    )

    # train
    trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/results_thesis/final_ray/last.ckpt')
    # trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/other_re_th/last.ckpt')
    # trainer.fit(net, datamodule=openkbp)

    return net


if __name__ == '__main__':
    net_ = main()