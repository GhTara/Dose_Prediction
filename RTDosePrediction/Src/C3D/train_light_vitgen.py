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

import bitsandbytes as bnb

from typing import Optional

from RTDosePrediction.Src.C3D.model_ablation import *
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
    def __init__(self, crop_flag, image_size):
        super().__init__()
        self.crop_flag = crop_flag
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_data = get_dataset(path=config.MAIN_PATH + config.TRAIN_DIR, state='train',
                                      size=100, cache=True, crop_flag=self.crop_flag, image_size=self.image_size)

        self.val_data = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='val',
                                    size=20, cache=True)

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
                                    size=20, cache=True)
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
            image_size=128,
            sw_batch_size=1,
            huber=False,
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
        
        # change size of the image: 96
        self.model_ = VitGenerator(
            in_ch=9,
            out_ch=1,
            mode_decoder=1,
            mode_encoder=1,
            feature_size=16,
            img_size=(image_size, image_size, image_size),
            num_layers=8,  # 4, 8, 12
            num_heads=6,  # 3, 6, 12,
            mode_multi_dec=True,
            # act='mish',
            act=config_param["act"],
            # multiS_conv=True,
            multiS_conv=config_param["multiS_conv"],)
        
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

        # self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_type = config_param["lr"]
        self.weight_decay = config_param["weight_decay"]

        self.loss_function = GenLoss(im_size=image_size)
        self.huber = huber
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

        self.image_size = image_size
        # self.img_width = image_size
        # self.img_depth = image_size
        
        self.sw_batch_size = sw_batch_size
        
        self.list_DVH_dif = []
        self.list_dose_metric = []
        self.dict_DVH_dif = {}
        self.ivs_values = []

    def forward(self, x):
        return self.model_(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch['Input'].float()
        target = batch['GT']

        # train
        output = self(input_)
        torch.cuda.empty_cache()

        loss = self.loss_function(output, target, casecade=False, huber=self.huber)

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

        roi_size = (self.image_size, self.image_size, self.image_size)
    
        # prediction = self.forward(input_)
        
        
        prediction = sliding_window_inference(
            input_, roi_size, self.sw_batch_size, lambda x: self.forward(x)[0]
        )
        torch.cuda.empty_cache()
        loss = self.loss_function(prediction, target, mode='val', huber=self.huber)

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
        # self.logger.log_hyperparams(params=dict(num_layers=self.config_param["num_layers"],  # 4, 8, 12
        #                                         num_heads=self.config_param["num_heads"]))
        optimizer = bnb.optim.Adam8bit(self.model_.parameters(), lr=self.lr_scheduler_type, weight_decay=self.weight_decay)
                                                
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
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        # return [optimizer], [scheduler]
        
    def test_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = target[:, :1, :, :, :].cpu()
        possible_dose_mask = target[:, 1:, :, :, :].cpu()

        roi_size = (self.image_size, self.image_size, self.image_size)
    
        # prediction = self.forward(input_)
        
        
        prediction = sliding_window_inference(
            input_, roi_size, self.sw_batch_size, lambda x: self.forward(x)[0]
        ).cpu()
        torch.cuda.empty_cache()

        # prediction_b = np.array(prediction.cpu())

        # Post-processing and evaluation
        prediction[np.logical_or(possible_dose_mask < 1, prediction < 0)] = 0
        prediction = 80. * prediction
        
        dose_dif, DVH_dif, self.dict_DVH_dif, ivs_values = get_Dose_score_and_DVH_score_batch(
            prediction, batch_data, list_DVH_dif=self.list_DVH_dif, dict_DVH_dif=self.dict_DVH_dif, ivs_values=None)
        self.list_DVH_dif.append(DVH_dif)
        
        torch.cuda.empty_cache()
        # if False:
        ckp_re_dir = '/content/drive/MyDrive/results_thesis_images/ablation3'
        if batch_idx<12:
            plot_DVH(prediction, batch_data, path=os.path.join(ckp_re_dir, 'dvh_{}.png'.format(batch_idx)))
        
            prediction_b = prediction.cpu()
            

            # Post-processing and evaluation
            # prediction[torch.logical_or(possible_dose_mask < 1, prediction < 0)] = 0
            # for save image
            gt_dose[possible_dose_mask < 1] = 0
            
            pred_img = torch.permute(prediction[0].cpu(), (1,0,2,3))
            gt_img = torch.permute(gt_dose[0], (1,0,2,3))
            name_p = batch_data['file_path'][0].split("/")[-2]
            
            for i in range(len(pred_img)):
                pred_i = pred_img[i][0].numpy()
                gt_i = 80. * gt_img[i][0].numpy()
                error = np.abs(gt_i - pred_i)
                
                # plt.figure("check", (12, 6))
                # Create a figure and axis object using Matplotlib
                fig, axs = plt.subplots(3, 1, figsize=(4, 10))
                plt.subplots_adjust(wspace=0, hspace=0)
            
                # Display the ground truth array
                axs[0].imshow(gt_i, cmap='jet')
                # axs[0].set_title('Ground Truth')
                axs[0].axis('off')
            
                # Display the prediction array
                axs[1].imshow(pred_i, cmap='jet')
                # axs[1].set_title('Prediction')
                axs[1].axis('off')
                
                # Display the error map using a heatmap
                heatmap = axs[2].imshow(error, cmap='jet')
                # axs[2].set_title('Error Map')
                axs[2].axis('off')
            
                save_dir = os.path.join(ckp_re_dir, '{}_{}'.format(name_p, batch_idx))
                if not os.path.isdir(save_dir):
                            os.mkdir(save_dir)
            
                fig.savefig(os.path.join(save_dir, '{}.jpg'.format(i)), bbox_inches="tight")
            
            # save as image
            # pred_grid = make_grid(pred_img, nrow=11)
            # gt_grid = make_grid(gt_img, nrow=11)
            
            # if not os.path.isdir(config.CHECKPOINT_RESULT_DIR):
            #     os.mkdir(config.CHECKPOINT_RESULT_DIR)
                
            # plt.imsave(os.path.join(config.CHECKPOINT_RESULT_DIR, 'gt_{}.jpg'.format(batch_idx)), gt_grid[0, :, :])
            # plt.imsave(os.path.join(config.CHECKPOINT_RESULT_DIR, 'pred_{}.jpg'.format(batch_idx)), pred_grid[0, :, :])
            # save_image(pred_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'pred_{}.jpg'.format(batch_idx)))
            # save_image(gt_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'gt_{}.jpg'.format(batch_idx)))
        
        self.list_dose_metric.append(dose_dif)
        return {"dose_dif": dose_dif}


    def test_epoch_end(self, outputs):
        
        mean_dose_metric = np.stack([x["dose_dif"] for x in outputs]).mean()
        std_dose_metric = np.stack([x["dose_dif"] for x in outputs]).std()
        print(np.min(std_dose_metric))
        # mean_dvh_metric = torch.stack(self.list_DVH_dif).mean()
        mean_dvh_metric = np.mean(self.list_DVH_dif)
        std_dvh_metric = np.std(self.list_DVH_dif)
        print(np.min(self.list_DVH_dif))
        # for key in self.dict_DVH_dif.keys():
        #     for metric in self.dict_DVH_dif[key]:
        #         self.dict_DVH_dif[key][metric] = np.mean(self.dict_DVH_dif[key][metric])
        
        print(mean_dose_metric, mean_dvh_metric)
        print('----------------------Difference DVH for each structures---------------------')
        print(self.dict_DVH_dif)
        # print('----------------------predicted---------------------')
        # print(self.pred_list_DVH)
        
        self.ivs_values = np.array(self.ivs_values)

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("std_dose_metric", std_dose_metric)
        self.log("mean_dvh_metric", mean_dvh_metric)
        self.log("std_dvh_metric", std_dvh_metric)
        # self.logger.log_metrics({"mean_dose_score": mean_dose_score}, self.current_epoch + 1)
        # self.logger.log_metrics({"val_loss": avg_loss}, self.current_epoch + 1)
        return self.dict_DVH_dif
        


def main(act, crop_flag, sw_batch_size, image_size, huber, resum, databricks, ckp):
    # initialise the LightningModule
    openkbp = OpenKBPDataModule(crop_flag=crop_flag, image_size=image_size)
    config_param = {
        "num_layers": 8,
        "num_heads": 6,
        "act": act,
        "multiS_conv": True,
        "lr": 3e-4,
        "weight_decay": 1e-4,
    }
    net = CascadeUNet(
        config_param,
        lr_scheduler_type='cosine',
        lr=3e-4,
        weight_decay=1e-4,
        eta_min=1e-7,
        last_epoch=-1,
        image_size=image_size, 
        huber=huber,
        sw_batch_size=sw_batch_size,
    )

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckp,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
        #  filename=net.filename,
    )

    # set up logger
    if not resum:
        mlflow_logger = MLFlowLogger(
            experiment_name='/Users/gheshlaghitara@gmail.com/dose_prediction_vitGen',
            tracking_uri="databricks",
            run_name = databricks
        )
    else:
        mlflow_logger = MLFlowLogger(
                experiment_name='/Users/gheshlaghitara@gmail.com/dose_prediction_vitGen',
                tracking_uri="databricks",
                run_id = databricks
            )
    

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        # callbacks=RichProgressBar(),
        # callbacks=[bar],
        default_root_dir=ckp,
        # enable_progress_bar=True,
        # log_every_n_steps=net.check_val,
    )

    # train
    # trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/results_thesis/ablation1_dose/last.ckpt')
    # trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/other_re_th/last.ckpt')
    if not resum:
        trainer.fit(net, datamodule=openkbp)
    else:
        trainer.fit(net, datamodule=openkbp, ckpt_path=os.path.join(ckp, 'last.ckpt'))
        
    return net


if __name__ == '__main__':
    net_ = main()