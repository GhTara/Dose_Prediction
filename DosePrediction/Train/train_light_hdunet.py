import DosePrediction.Train.config as config
import gc
from typing import Optional

from monai.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from DosePrediction.Models.Networks.hdunet import *
from DosePrediction.DataLoader.dataloader_OpenKBP_monai import get_dataset
from DosePrediction.Evaluate.evaluate_openKBP import *
from DosePrediction.Train.loss import Loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


class OpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

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


class HDUNet(pl.LightningModule):
    def __init__(
            self,
            config_param,
            freez=False,
    ):
        super().__init__()
        self.config_param = config_param
        self.save_hyperparameters()

        # OAR + PTV + CT => dose

        self.model_ = Model(in_ch=9, growth_rate=16, upsample_chan=64)


        self.loss_function = Loss(False)
        # Moving average loss, loss is the smaller the better
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
        self.freez = freez

    def forward(self, x):
        return self.model_(x)[0]

    def training_step(self, batch, batch_idx):
        input_ = batch['Input'].float()
        target = batch['GT']

        # train
        output = self(input_)
        torch.cuda.empty_cache()

        loss = self.loss_function(output, target, freez=self.freez)

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

        loss = self.loss_function(prediction, target, freez=self.freez)

        prediction_b = np.array(prediction.cpu())

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


def main(ckpt_path):
    # initialise the LightningModule
    openkbp = OpenKBPDataModule()
    config_param = {
        "num_layers": 4,
        "num_heads": 6,
    }
    net = HDUNet(
        config_param,
    )

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
    )

    # set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='EXPERIMENT_NAME',
        tracking_uri="databricks",
        run_id="run_id"
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
    trainer.fit(net, datamodule=openkbp,
                ckpt_path=os.path.join(ckpt_path, 'last.ckpt'))

    return net


if __name__ == '__main__':
    net_ = main()
