import sys
import random

sys.path.insert(0, '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/')

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.networks import normal_init, icnr_init

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, ProgressBarBase
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from ray_lightning import RayShardedStrategy

from torch.nn.functional import interpolate
from torch.nn import init

from RTDosePrediction.Src.C3D.my_model import *
from RTDosePrediction.Src.DataLoader.dataloader_OpenKBP_C3D_monai import get_dataset
import RTDosePrediction.Src.DataLoader.config as config
from RTDosePrediction.Src.Evaluate.evaluate_openKBP import *
from RTDosePrediction.Src.C3D.loss import Loss, GenLoss, DiscLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
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
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function  < init_func>


def seed_everything(seed=33):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


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


class GANMultiDisc(pl.LightningModule):
    
    def __init__(
            self,
            config_param,
            feature_size,
            delta3=2,
            # Adam hp
            G_lr: float = 10e-5,
            D_lr: float = 4*10e-4,
            weight_decay: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            init_type='kaiming', 
            init_gain=0.02,
            std_noise=0.1,
            init_w=False,
            resnet_disc=False,
    ):
        super().__init__()
        self.config_param = config_param
        self.delta3 = delta3
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.init_type = init_type
        self.init_gain= init_gain
        self.std_noise = std_noise
        
        self.save_hyperparameters()
        seed_everything()

        self.generator = VitGenerator(
            in_ch=9,
            out_ch=1,
            mode_decoder=1,
            mode_encoder=1,
            feature_size=feature_size,
            img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
            num_layers=8,  # 4, 8, 12
            num_heads=6  # 3, 6, 12
        ).to(torch.device('cuda'))
    

        # self.discriminator = Regressor((10, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
        #                               channels=(feature_size * 2, feature_size * 4, feature_size * 8, 1),
        #                               strides=(2, 2, 2, 1), padding=(1, 1, 1, 1),
        #                               kernel_size=(4, 4, 4, 3), num_res_units=0).to(torch.device('cuda'))
                                       
        self.discriminator = AttRegressor((10, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
        channels=(feature_size, feature_size * 2, feature_size * 4, feature_size * 8, 1), 
        strides=(1, 2, 2, 2, 1), padding=(1, 1, 1, 1, 1),
        kernel_size=(3, 4, 4, 4, 3), num_res_units=0, 
        std_noise=std_noise).to(torch.device('cuda'))
        

        if init_w:
            init_weights(self.generator, init_type=self.init_type)
            init_weights(self.discriminator, init_type=self.init_type)

        self.recon_criterion = GenLoss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()

        self.eps_train_loss = 0.01

        self.best_average_val_index = -99999999.
        self.average_val_index = None
        self.metric_values = []

        self.best_average_train_loss = 99999999.
        self.moving_train_loss = None
        self.train_epoch_loss = []

        self.max_epochs = 1300
        # self.max_epochs = 150
        self.check_val = 4
        self.warmup_epochs = 2


    
    def forward(self, x):
        return self.generator(x)
        
    def generate_multi_scale(self, vol):
        volumes = [vol]
        ch = vol.shape[-1]
        # 4 is depth
        for i in range(1, 4):
            dim = config.IMAGE_SIZE // np.power(2, i)
            volume_int = interpolate(vol, size=(dim, dim, dim), mode='trilinear', align_corners=True)
            volumes.append(volume_int)
        return volumes

    def training_step(self, batch, batch_idx, optimizer_idx):
        conditioned_image = batch['Input'].float()
        real_image = batch['GT']

        # Train generator
        if optimizer_idx == 0:
            fake_image = self(conditioned_image)
            fake_disc_logits = self.discriminator(fake_image, conditioned_image)
            real_disc_logits = self.discriminator(self.generate_multi_scale(real_image[:, 0:1, :, :, :]), conditioned_image)
            
            torch.cuda.empty_cache()
            
            # Adversarial loss is binary cross-entropy
            adversarial_loss = self.adversarial_criterion(fake_disc_logits, torch.ones_like(fake_disc_logits))
            # Voxel-wise loss
            dose_loss = self.recon_criterion(fake_image, real_image)
            # Total loss
            g_loss = self.delta3 * adversarial_loss + dose_loss

            return {"loss": g_loss, "optimizer_idx": optimizer_idx}

        # Train discriminator
        if optimizer_idx == 1:
            real_disc_logits = self.discriminator(self.generate_multi_scale(real_image[:, 0:1, :, :, :]), conditioned_image)
            fake_image = self(conditioned_image)
            fake_disc_logits = self.discriminator(fake_image, conditioned_image)
            torch.cuda.empty_cache()

            d_loss = self.adversarial_criterion(real_disc_logits, torch.ones_like(real_disc_logits)) + \
                     self.adversarial_criterion(fake_disc_logits, torch.zeros_like(fake_disc_logits))

            return {"loss": d_loss, "optimizer_idx": optimizer_idx}

    def training_epoch_end(self, outputs):
        gan_mean_loss = torch.stack([x[0]["loss"] for x in outputs]).mean()
        self.logger.log_metrics({"gan_loss": gan_mean_loss}, self.current_epoch + 1)
        disc_mean_loss = torch.stack([x[1]["loss"] for x in outputs]).mean()
        self.logger.log_metrics({"disc_loss": disc_mean_loss}, self.current_epoch + 1)

    def validation_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = np.array(target[:, :1, :, :, :].cpu())
        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        sw_batch_size = 1
        prediction = self.forward(input_)
        torch.cuda.empty_cache()
        # fake_disc_logits = self.discriminator(prediction)

        # Adversarial loss is binary cross-entropy
        # adversarial_loss = self.adversarial_criterion(fake_disc_logits, torch.ones_like(fake_disc_logits))
        # Voxel-wise loss
        dose_loss = self.recon_criterion(prediction, target)
        # Total loss
        # g_loss = self.delta3 * adversarial_loss + dose_loss
        # prediction_b = sliding_window_inference(
        #     input_, roi_size, sw_batch_size, lambda x: self.forward(x)[1]
        # )
        # prediction_b = np.array(prediction[1].cpu())
        # for vit generator
        prediction_b = np.array(prediction[0].cpu())

        # Post-processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction_b < 0)
        prediction_b[mask] = 0
        dose_score = 80. * get_3D_Dose_dif(prediction_b.squeeze(0), gt_dose.squeeze(0),
                                           possible_dose_mask.squeeze(0))

        return {"val_loss": dose_loss, "val_metric": dose_score}

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
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.G_lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.D_lr)
        return [opt_g, opt_d]


def main():
    # initialise the LightningModule
    openkbp = OpenKBPDataModule()
    config_param = {
        "num_layers": 4,
        "num_heads": 6,
        # "lr": tune.loguniform(1e-4, 1e-1),
    }
    net = GANMultiDisc(
        config_param,
        feature_size=16,
        std_noise=0.1,
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
        experiment_name='/Users/gheshlaghitara@gmail.com/dose_prediction_GAN',
        tracking_uri="databricks",
        run_id = 'fea9e68a46294231ab7e83c64a3d5d24'
    )

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
    trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/results_thesis/re_swin_unetr/last.ckpt')
    # trainer.fit(net, datamodule=openkbp, ckpt_path='/content/drive/MyDrive/other_re_th/last.ckpt')

    return net


if __name__ == '__main__':
    net_ = main()
