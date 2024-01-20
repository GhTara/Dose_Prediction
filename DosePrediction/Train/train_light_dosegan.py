import DosePrediction.Train.config as config
import gc
from typing import Optional

from monai.data import DataLoader, list_data_collate

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from DosePrediction.Models.Networks.dosegan import *
from DosePrediction.DataLoader.dataloader_OpenKBP_monai import get_dataset
from DosePrediction.Evaluate.evaluate_openKBP import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True


class OpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.val_data = None

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
                                     size=100, cache=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)


class DoseGAN(pl.LightningModule):
    def __init__(
            self,
            config_param,
            freeze=True
    ):
        super().__init__()
        self.config_param = config_param
        self.freeze = freeze
        self.save_hyperparameters()
        self.isTrain = config_param["isTrain"]

        # load/define networks
        device = torch.device("cuda")
        self.netG = UnetGenerator3d(input_nc=9, output_nc=1, num_downs=6, ngf=64,
                                    norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]).to(device)
        if self.isTrain:
            self.netD = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3).to(device)

        self.delta3 = config_param["delta3"]
        self.G_lr = config_param["G_lr"]
        self.D_lr = config_param["D_lr"]

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.eps_train_loss = 0.01

        self.best_average_val_index = -99999999.
        self.average_val_index = None
        self.metric_values = []

        self.best_average_train_loss = 99999999.
        self.moving_train_loss = None
        self.train_epoch_loss_gan = []
        self.train_epoch_loss_disc = []

        self.max_epochs = 1300
        self.check_val = 5
        self.warmup_epochs = 1

        self.img_height = config.IMAGE_SIZE
        self.img_width = config.IMAGE_SIZE
        self.img_depth = config.IMAGE_SIZE

        self.sw_batch_size = config.SW_BATCH_SIZE

        self.list_DVH_dif = []
        self.list_dose_metric = []
        self.dict_DVH_dif = {}
        self.ivs_values = []

    def forward(self, x):
        return self.netG.forward(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        torch.cuda.empty_cache()
        input_ = batch['Input'].float()
        real_out = batch['GT'][:, :1, :, :, :]

        # Train generator
        if optimizer_idx == 0:
            fake_output = self(input_)
            pred_fake = self.netD(fake_output)
            loss_g_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake).type_as(pred_fake))

            loss_g_l1 = self.criterionL1(fake_output, real_out) * self.delta3

            torch.cuda.empty_cache()

            g_loss = loss_g_gan + loss_g_l1

            return {"loss": g_loss, "optimizer_idx": optimizer_idx}

        # Train discriminator
        if optimizer_idx == 1:
            pred_fake = self.netD(self(input_).detach())
            d_loss_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake).type_as(pred_fake))

            pred_real = self.netD(real_out)
            d_loss_real = self.criterion_gan(pred_real, torch.ones_like(pred_real).type_as(pred_real))

            d_loss = (d_loss_fake + d_loss_real) * 0.5

            torch.cuda.empty_cache()

            return {"loss": d_loss, "optimizer_idx": optimizer_idx}

    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()

        gan_mean_loss = torch.stack([x[0]["loss"] for x in outputs]).mean()
        self.train_epoch_loss_gan.append(gan_mean_loss.detach().cpu().numpy())
        self.logger.log_metrics({"gan_loss": gan_mean_loss}, self.current_epoch + 1)

        disc_mean_loss = torch.stack([x[1]["loss"] for x in outputs]).mean()
        self.train_epoch_loss_disc.append(disc_mean_loss.detach().cpu().numpy())
        self.logger.log_metrics({"disc_loss": disc_mean_loss}, self.current_epoch + 1)

        torch.cuda.empty_cache()

    def validation_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]
        gt_dose = target[:, :1, :, :, :]
        possible_dose_mask = target[:, 1:, :, :, :]

        prediction = self.forward(input_)

        torch.cuda.empty_cache()

        loss = self.criterionL1(prediction, gt_dose)
        gt_dose = np.array(gt_dose.cpu())
        possible_dose_mask = np.array(possible_dose_mask.cpu())

        prediction_b = np.array(prediction.cpu())

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
        opt_g = torch.optim.Adam(self.netG.parameters(),
                                 lr=self.G_lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.netD.parameters(),
                                 lr=self.D_lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]

    def test_step(self, batch_data, batch_idx):
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]

        possible_dose_mask = np.array(target[:, 1:, :, :, :].cpu())

        prediction = np.array(self.forward(input_).cpu())

        # Post-processing and evaluation
        mask = np.logical_or(possible_dose_mask < 1, prediction < 0)
        prediction[mask] = 0

        # To save image
        prediction = 70. * prediction

        dose_dif, DVH_dif, self.dict_DVH_dif, ivs_values = get_Dose_score_and_DVH_score_batch(
            prediction, batch_data, list_DVH_dif=self.list_DVH_dif, dict_DVH_dif=self.dict_DVH_dif, ivs_values=None)
        self.list_DVH_dif.append(DVH_dif)
        torch.cuda.empty_cache()
        # if False:
        ckp_re_dir = os.path.join('YourSampleImages/DosePrediction' + 'ours_model')
        if batch_idx < 100:
            plot_DVH(prediction, batch_data, path=os.path.join(ckp_re_dir, 'dvh_{}.png'.format(batch_idx)))
            torch.cuda.empty_cache()
            del batch_data
            del prediction
            gc.collect()
        # Make it True if you want the output as images
        if False:
            prediction_b = prediction.cpu()

            # Post-processing and evaluation
            # prediction[torch.logical_or(possible_dose_mask < 1, prediction < 0)] = 0
            # for save image
            gt_dose[possible_dose_mask < 1] = 0

            pred_img = torch.permute(prediction[0].cpu(), (1, 0, 2, 3))
            gt_img = torch.permute(gt_dose[0], (1, 0, 2, 3))
            name_p = batch_data['file_path'][0].split("/")[-2]

            for i in range(len(pred_img)):
                pred_i = pred_img[i][0].numpy()
                gt_i = 70. * gt_img[i][0].numpy()
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

        self.list_dose_metric.append(dose_dif)
        return {"dose_dif": dose_dif}

    def test_epoch_end(self, outputs):

        mean_dose_metric = np.stack([x["dose_dif"] for x in outputs]).mean()
        std_dose_metric = np.stack([x["dose_dif"] for x in outputs]).std()
        mean_dvh_metric = np.mean(self.list_DVH_dif)
        std_dvh_metric = np.std(self.list_DVH_dif)

        print(mean_dose_metric, mean_dvh_metric)
        print('----------------------Difference DVH for each structures---------------------')
        print(self.dict_DVH_dif)

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("std_dose_metric", std_dose_metric)
        self.log("mean_dvh_metric", mean_dvh_metric)
        self.log("std_dvh_metric", std_dvh_metric)


def main(freeze=True, isTrain=True, delta3=10, run_id=None, run_name=None, ckpt_path=None, lr=0.0002):
    # initialise the LightningModule
    openkbp = OpenKBPDataModule()
    config_param = {
        'delta3': delta3,
        "G_lr": lr,
        "D_lr": lr,
        "isTrain": isTrain
    }
    net = DoseGAN(
        config_param,
        freeze=freeze
    )

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
    )

    # set up logger
    if run_name is None:
        mlflow_logger = MLFlowLogger(
            experiment_name='EXPERIMENT_NAME',
            tracking_uri="databricks",
            run_id=run_id
        )
    else:
        mlflow_logger = MLFlowLogger(
            experiment_name='EXPERIMENT_NAME',
            tracking_uri="databricks",
            run_name=run_name
        )

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        default_root_dir=ckpt_path,
    )

    # train
    if run_name is None:
        # To load the model for further training or inference
        loaded_gan_model = DoseGAN.load_from_checkpoint(
            checkpoint_path=os.path.join(ckpt_path, 'last.ckpt'),
            netG=net.netG,  # pass the generator
            netD=net.netD,  # pass the discriminator
        )

        # Continue training or perform inference with the loaded model
        trainer.fit(loaded_gan_model, datamodule=openkbp, ckpt_path=os.path.join(ckpt_path, 'last.ckpt'))
    else:
        trainer.fit(net, datamodule=openkbp)

    return net


if __name__ == '__main__':
    net_ = main()
