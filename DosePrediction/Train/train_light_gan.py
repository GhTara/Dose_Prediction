from typing import Optional

from monai.networks.nets import resnet10

import DosePrediction.Train.config as config

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from torch.nn import init

import bitsandbytes as bnb

from DosePrediction.Models.Networks.dose_pyfer import *
from DosePrediction.DataLoader.dataloader_OpenKBP_monai import get_dataset
from DosePrediction.Evaluate.evaluate_openKBP import *
from DosePrediction.Models.Networks.models_experiments import create_pretrained_medical_resnet
from DosePrediction.Train.loss import GenLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper.
    """

    def init_func(m):  # define the initialization function
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif class_name.find(
                'BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function  < init_func>


class OpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.val_data = None
        self.train_data = None

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


class FineTuneCB(Callback):
    # add callback to freeze/unfreeze trained layers
    def __init__(self, unfreeze_epoch: int) -> None:
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        for n, param in pl_module.discriminator.named_parameters():
            param.requires_grad = True
        optimizers = pl_module.configure_optimizers()
        trainer.optimizers = optimizers


class GAN(pl.LightningModule):

    def __init__(
            self,
            config_param,
            delta3=2,
            # Adam hp
            G_lr: float = 10e-5,
            D_lr: float = 5 * 10e-4,
            init_type='kaiming',
            init_gain=0.02,
            std_noise=0.1,
            init_w=False,
    ):
        super().__init__()
        self.config_param = config_param
        self.delta3 = delta3
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.init_type = init_type
        self.init_gain = init_gain
        self.std_noise = std_noise

        self.save_hyperparameters()

        self.generator = MainSubsetModel(
            in_ch=9,
            out_ch=1,
            feature_size=16,
            img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
            num_layers=8,
            num_heads=6,
            mode_multi_dec=True,
            act='mish',
            multiS_conv=False
        )

        self.discriminator, self.pretrained_params = create_pretrained_medical_resnet(
            'PretrainedModels/resnet_10_23dataset.pth', model_constructor=resnet10,
            num_classes=2, n_input_channels=1, spatial_dims=3)

        self.pretrained_params = set(self.pretrained_params)
        for n, param in self.discriminator.named_parameters():
            param.requires_grad = bool(n not in self.pretrained_params)

        if init_w:
            init_weights(self.generator, init_type=self.init_type)
            init_weights(self.discriminator, init_type=self.init_type)

        self.recon_criterion = GenLoss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()

        self.eps_train_loss = 0.01

        self.best_average_val_index = -99999999.
        self.average_val_index = None

        self.best_average_train_loss = 99999999.
        self.moving_train_loss = None
        self.train_epoch_loss = []

        self.max_epochs = 1300
        # self.max_epochs = 150
        self.check_val = 1
        self.warmup_epochs = 1

        self.sw_batch_size = config.SW_BATCH_SIZE

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        torch.cuda.empty_cache()
        conditioned_image = batch['Input'].float()
        real_image = batch['GT']

        # Train generator
        if optimizer_idx == 0:
            fake_image = self(conditioned_image)
            fake_disc_logit = self.discriminator(fake_image[0])
            real_disc_logit = self.discriminator(real_image[:, 0:1, :, :, :])

            torch.cuda.empty_cache()

            # Adversarial loss is binary cross-entropy
            adversarial_loss = self.adversarial_criterion(fake_disc_logit, torch.ones_like(fake_disc_logit))
            # Voxel-wise loss
            dose_loss = self.recon_criterion(fake_image, real_image)
            # Total loss
            g_loss = self.delta3 * adversarial_loss + dose_loss

            return {"loss": g_loss, "optimizer_idx": optimizer_idx}

        # Train discriminator
        if optimizer_idx == 1:
            real_disc_logit = self.discriminator(real_image[:, 0:1, :, :, :])
            fake_image = self(conditioned_image)
            fake_disc_logit = self.discriminator(fake_image[0])
            torch.cuda.empty_cache()

            d_loss = 0.5 * (self.adversarial_criterion(real_disc_logit, torch.ones_like(real_disc_logit)) +
                            self.adversarial_criterion(fake_disc_logit, torch.zeros_like(fake_disc_logit)))

            return {"loss": d_loss, "optimizer_idx": optimizer_idx}

    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        gan_mean_loss = torch.stack([x[0]["loss"] for x in outputs]).mean()
        self.logger.log_metrics({"gan_loss": gan_mean_loss}, self.current_epoch + 1)
        disc_mean_loss = torch.stack([x[1]["loss"] for x in outputs]).mean()
        self.logger.log_metrics({"disc_loss": disc_mean_loss}, self.current_epoch + 1)
        torch.cuda.empty_cache()

    def validation_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)

        torch.cuda.empty_cache()
        prediction = sliding_window_inference(
            input_, roi_size, self.sw_batch_size, lambda x: self.forward(x)[0]
        )
        torch.cuda.empty_cache()
        dose_loss = self.recon_criterion(prediction, target, mode='val')

        # For generator
        prediction_b = np.array(prediction.cpu())

        # Post-processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_b < 0)
        prediction_b[mask] = 0
        dose_score = 80. * get_3D_Dose_dif(prediction_b.squeeze(0), gt_dose.squeeze(0),
                                           possible_dose_mask.squeeze(0))
        torch.cuda.empty_cache()

        return {"val_loss": dose_loss, "val_metric": dose_score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_dose_score = - np.stack([x["val_metric"] for x in outputs]).mean()

        self.log("mean_dose_score", mean_dose_score, logger=False)
        self.log("val_loss", avg_loss, logger=False)
        self.logger.log_metrics({"mean_dose_score": mean_dose_score}, self.current_epoch + 1)
        self.logger.log_metrics({"val_loss": avg_loss}, self.current_epoch + 1)

    def configure_optimizers(self):
        opt_g = bnb.optim.Adam8bit(self.generator.parameters(), lr=self.G_lr)
        opt_d = bnb.optim.Adam8bit(self.discriminator.parameters(), lr=self.D_lr)
        return [opt_g, opt_d]


def main():
    # Initialise the LightningModule
    openkbp_ds = OpenKBPDataModule()
    config_param = {
        "num_layers": 4,
        "num_heads": 6,
    }
    net = GAN(
        config_param,
        std_noise=0.1,
    )

    # Set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_MODEL_DIR_DOSE_GAN,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
    )

    # Set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='EXPERIMENT_NAME',
        tracking_uri="databricks",
        run_name='RUN_NAME'
        # run_id = 'RUN_ID'
    )

    fine = FineTuneCB(unfreeze_epoch=10)

    # Initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback, fine],
        logger=mlflow_logger,
        default_root_dir=config.CHECKPOINT_MODEL_DIR_DOSE_GAN,
        # enable_progress_bar=True,
        # log_every_n_steps=net.check_val,
    )

    # Train
    # trainer.fit(net,
    # datamodule=openkbp_ds,
    # ckpt_path=os.path.join(config.CHECKPOINT_MODEL_DIR_DOSE_GAN, 'last.ckpt'))
    trainer.fit(net, datamodule=openkbp_ds)

    return net


if __name__ == '__main__':
    net_ = main()
