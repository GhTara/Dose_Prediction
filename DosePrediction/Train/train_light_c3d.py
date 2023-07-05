import DosePrediction.Train.config as config
import gc

from monai.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from typing import Optional

from DosePrediction.Models.Networks.c3d import *
from DosePrediction.DataLoader.dataloader_OpenKBP_monai import get_dataset
from DosePrediction.Evaluate.evaluate_openKBP import *
from DosePrediction.Train.loss import Loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


class OpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.val_data = None
        self.train_data = None

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_data = get_dataset(path=config.MAIN_PATH + config.TRAIN_DIR, state='train',
                                      size=200, cache=True)

        self.val_data = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='val',
                                    size=100, cache=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=config.NUM_WORKERS, pin_memory=True)

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


class C3D(pl.LightningModule):
    def __init__(
            self,
            config_param,
            lr_scheduler_type='cosine',
            lr=3e-4,
            weight_decay=1e-4,
            eta_min=1e-7,
            last_epoch=-1,
            freeze=True,
    ):
        super().__init__()
        self.config_param = config_param
        self.save_hyperparameters()

        # OAR + PTV + CT => dose
        self.model_, inside = create_pretrained_unet(
            in_ch=9, out_ch=1,
            list_ch_A=[-1, 16, 32, 64, 128, 256],
            list_ch_B=[-1, 32, 64, 128, 256, 512],
            ckpt_file='HOME_DIRECTORY' + '/PretrainedModels/baseline/C3D_bs4_iter80000.pkl', )

        if freeze:
            for n, param in self.model_.named_parameters():
                if 'net_A' in n or 'conv_out_A' in n:
                    param.requires_grad = False
        self.freeze = freeze

        self.lr_scheduler_type = lr_scheduler_type

        self.loss_function = Loss()
        # Moving average loss, loss is the smaller, the better
        self.eps_train_loss = 0.01

        self.best_average_val_index = -99999999.
        self.average_val_index = None
        self.metric_values = []

        self.best_average_train_loss = 99999999.
        self.moving_train_loss = None
        self.train_epoch_loss = []

        self.max_epochs = 1300
        self.check_val = 10
        self.warmup_epochs = 5

        self.img_height = config.IMAGE_SIZE
        self.img_width = config.IMAGE_SIZE
        self.img_depth = config.IMAGE_SIZE

        self.list_DVH_dif = []
        self.list_dose_metric = []
        self.dict_DVH_dif = {}

    def forward(self, x):
        return self.model_(x)

    def training_step(self, batch, batch_idx):
        input_ = batch['Input'].float()
        target = batch['GT']

        output = self(input_)
        torch.cuda.empty_cache()

        loss = self.loss_function(output, target, freez=self.freeze)

        if self.moving_train_loss is None:
            self.moving_train_loss = loss.item()
        else:
            self.moving_train_loss = \
                (1 - self.eps_train_loss) * self.moving_train_loss \
                + self.eps_train_loss * loss.item()

        tensorboard_logs = {"train_loss": loss.item()}

        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if train_mean_loss < self.best_average_train_loss:
            self.best_average_train_loss = train_mean_loss
        self.train_epoch_loss.append(train_mean_loss.detach().cpu().numpy())
        self.logger.log_metrics({"train_mean_loss": train_mean_loss}, self.current_epoch + 1)

    def validation_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        prediction = self.forward(input_)
        torch.cuda.empty_cache()

        loss = self.loss_function(prediction, target, freez=self.freeze)

        prediction_b = np.array(prediction[1].cpu())

        # Post-processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_b < 0)
        prediction_b[mask] = 0
        dose_score = 70. * get_3D_Dose_dif(prediction_b.squeeze(0), gt_dose.squeeze(0),
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

    def configure_optimizers(self):
        lr = self.hparams.lr
        lr_encoder = self.hparams.lr_encoder
        lr_decoder = self.hparams.lr_decoder
        weight_decay = self.hparams.weight_decay

        t_max = self.max_epochs
        eta_min = self.hparams.eta_min

        milestones = self.hparams.milestones
        gamma = self.hparams.gamma

        factor = self.hparams.factor
        patience = self.hparams.patience
        threshold = self.hparams.threshold

        last_epoch = self.hparams.last_epoch

        scheduler = None

        if hasattr(self.model_, 'decoder') and hasattr(self.model_, 'encoder'):
            optimizer = torch.optim.Adam([
                {'params': self.model_.encoder.parameters(), 'lr': lr_encoder},
                {'params': self.model_.decoder.parameters(), 'lr': lr_decoder}
            ],
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True)
        else:
            optimizer = torch.optim.Adam(self.model_.parameters(),
                                         lr=lr,
                                         weight_decay=3e-5,
                                         betas=(0.9, 0.999),
                                         eps=1e-08,
                                         amsgrad=True)

        if self.lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=t_max,
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
            return [optimizer], [{"scheduler": scheduler,
                                  "monitor": "moving_train_loss",
                                  }]

        return [optimizer], [{"scheduler": scheduler}]

    def test_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]

        gt_dose = target[:, :1, :, :, :].cpu()
        possible_dose_mask = target[:, 1:, :, :, :].cpu()

        prediction = self.forward(input_)[1].cpu()
        prediction[np.logical_or(possible_dose_mask < 1, prediction < 0)] = 0

        prediction = 70. * prediction

        dose_dif, DVH_dif, self.dict_DVH_dif, ivs_values = get_Dose_score_and_DVH_score_batch(
            prediction, batch_data, list_DVH_dif=self.list_DVH_dif, dict_DVH_dif=self.dict_DVH_dif, ivs_values=None)
        self.list_DVH_dif.append(DVH_dif)
        torch.cuda.empty_cache()

        save_results = True
        if save_results and batch_idx:
            ckp_re_dir = os.path.join('YourSampleImages/DosePrediction', 'baseline')
            plot_DVH(prediction, batch_data, path=os.path.join(ckp_re_dir, 'dvh_{}.png'.format(batch_idx)))
            # Post-processing and evaluation
            gt_dose[possible_dose_mask < 1] = 0

            predicted_img = torch.permute(prediction[0].cpu(), (1, 0, 2, 3))
            gt_img = torch.permute(gt_dose[0], (1, 0, 2, 3))
            name_p = batch_data['file_path'][0].split("/")[-2]

            for i in range(len(predicted_img)):
                predicted_i = predicted_img[i][0].numpy()
                gt_i = 70. * gt_img[i][0].numpy()
                error = np.abs(gt_i - predicted_i)

                # Create a figure and axis object using Matplotlib
                fig, axs = plt.subplots(3, 1, figsize=(4, 10))
                plt.subplots_adjust(wspace=0, hspace=0)

                # Display the ground truth array
                axs[0].imshow(gt_i, cmap='jet')
                # axs[0].set_title('Ground Truth')
                axs[0].axis('off')

                # Display the prediction array
                axs[1].imshow(predicted_i, cmap='jet')
                # axs[1].set_title('Prediction')
                axs[1].axis('off')

                # Display the error map using a heatmap
                axs[2].imshow(error, cmap='jet')
                # axs[2].set_title('Error Map')
                axs[2].axis('off')

                save_dir = os.path.join(ckp_re_dir, '{}_{}'.format(name_p, batch_idx))
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)

                fig.savefig(os.path.join(save_dir, '{}.jpg'.format(i)), bbox_inches="tight")

                torch.cuda.empty_cache()
                del batch_data
                del prediction
                gc.collect()

        self.list_dose_metric.append(dose_dif)
        return {"dose_dif": dose_dif}

    def test_epoch_end(self, outputs):

        mean_dose_metric = np.stack([x["dose_dif"] for x in outputs]).mean()
        std_dose_metric = np.stack([x["dose_dif"] for x in outputs]).std()
        mean_dvh_metric = np.mean(self.list_DVH_dif)

        print(mean_dose_metric, mean_dvh_metric)
        print('----------------------Difference DVH for each structures---------------------')
        print(self.dict_DVH_dif)

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("std_dose_metric", std_dose_metric)


def main(freeze=True):
    # Initialise the LightningModule
    openkbp_ds = OpenKBPDataModule()
    config_param = {
        "num_layers": 4,
        "num_heads": 6,
    }
    net = C3D(
        config_param,
        lr_scheduler_type='cosine',
        lr=3e-4,
        weight_decay=1e-4,
        eta_min=1e-7,
        last_epoch=-1,
        freeze=freeze
    )

    # Set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_MODEL_DIR_BASE_FINAL,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
    )

    # Set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='EXPERIMENT_NAME',
        tracking_uri="databricks",
        # run_name = 'RUN_NAME'
        run_id='RUN_ID'
    )

    # Initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        default_root_dir=config.CHECKPOINT_MODEL_DIR_BASE_FINAL,
        # enable_progress_bar=True,
        # log_every_n_steps=net.check_val,
    )

    # Train
    trainer.fit(net,
                datamodule=openkbp_ds,
                ckpt_path=os.path.join(config.CHECKPOINT_MODEL_DIR_BASE_FINAL, 'last.ckpt'))
    # trainer.fit(net, datamodule=openkbp_ds)

    return net


if __name__ == '__main__':
    net_ = main()
