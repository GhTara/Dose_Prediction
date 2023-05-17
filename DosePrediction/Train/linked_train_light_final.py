import sys
import gc

sys.path.insert(0, 'HOME_DIRECTORY')

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.metrics import HausdorffDistanceMetric

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
# from ray_lightning import RayShardedStrategy

from DosePrediction.Train.my_model import *
# from RTDosePrediction.DosePrediction.Train.model import *
from DosePrediction.DataLoader.couple_dataloader_OpenKBP_C3D_monai import get_dataset
import DosePrediction.Train.config as dose_config
from DosePrediction.Evaluate.evaluate_openKBP import *
from DosePrediction.Train.loss import GenLoss

from DosePrediction.Train.train_light_final import CascadeUNet
from OARSegmentation.train import litAutoSeg

import DosePrediction.Train.config as config
import OARSegmentation.config as config_seg

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

        self.test_data = get_dataset(path=dose_config.MAIN_PATH + dose_config.VAL_DIR, state='test',
                                     size=100, cache=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)


class CoupleNet(pl.LightningModule):
    def __init__(
            self,
            ckpt_file_path_dose,
            ckpt_file_path_seg,
            config_param,
            # Adam hp
            lr: float = 3e-4,
            weight_decay: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            freez=True,
    ):
        super().__init__()
        self.config_param = config_param
        self.freez = freez
        self.save_hyperparameters()

        # OAR + PTV + CT => dose
        self.dose_model = CascadeUNet(config_param=config_param)
        new_weights = self.dose_model.state_dict()
        old_weights = torch.load(ckpt_file_path_dose)['state_dict']

        new_keys = list(new_weights.keys())
        old_keys = list(old_weights.keys())
        for new_k, old_k in zip(new_keys, old_keys):
            new_weights[new_k] = old_weights[old_k]
        self.dose_model.load_state_dict(new_weights)

        self.seg_model = litAutoSeg(pretrain=False, mode_model=1)
        new_weights = self.seg_model.state_dict()
        old_weights = torch.load(ckpt_file_path_seg)['state_dict']

        new_keys = list(new_weights.keys())
        old_keys = list(old_weights.keys())
        for new_k, old_k in zip(new_keys, old_keys):
            new_weights[new_k] = old_weights[old_k]
        self.seg_model.load_state_dict(new_weights)

        self.lr_scheduler_type = config_param["lr"]
        self.weight_decay = config_param["weight_decay"]

        self.loss_function = GenLoss()
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

        self.sw_batch_size = config_seg.SW_BATCH_SIZE

        self.list_DVH_dif = []
        self.dict_DVH_dif = {}

        self.dice_metric_test = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.dice_metric_list = None

        self.hd95_metric_test = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean", get_not_nans=False
        )
        self.hd95_metric_list = None

    def forward1(self, x):
        return self.seg_model(x)

    def forward2(self, x):
        return self.dose_model(x)

    def test_step(self, batch_data, batch_idx):
        torch.cuda.empty_cache()
        input_ = batch_data["Input"].float()
        target = batch_data["GT"]

        ct = input_[:, :1, :, :, :]
        ptv = input_[:, 1:, :, :, :]

        gt1 = target[:, :1, :, :, :]
        gt2 = target[:, 1:, :, :, :]

        gt_dose = gt2[:, :1, :, :, :]
        possible_dose_mask = gt2[:, 1:, :, :, :].cpu()

        roi_size = (config_seg.IMAGE_SIZE, config_seg.IMAGE_SIZE, config_seg.IMAGE_SIZE)
        oars = sliding_window_inference(
            ct, roi_size, self.sw_batch_size, self.forward1)

        # size of batch is zero
        oars = [config_seg.post_pred(i) for i in decollate_batch(oars)][0]
        oars = torch.permute(oars, (0, 3, 2, 1))

        gt1 = [config_seg.post_label(i) for i in decollate_batch(gt1)][0]

        # ------------- to calculate metrics related to dose distribution predictor --------------
        ct = torch.permute(ct, (0, 1, 4, 3, 2))

        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        # sw_batch_size = 1
        oars = torch.unsqueeze(oars, dim=0)[:, 1:, :, :, :]

        structures = torch.cat((ptv, oars, ct), dim=1)
        prediction = self.forward2(structures)
        prediction = prediction[1][0].cpu()

        prediction[np.logical_or(possible_dose_mask < 1, prediction < 0)] = 0

        prediction = 70. * prediction

        dose_dif, DVH_dif, self.dict_DVH_dif, ivs_values = get_Dose_score_and_DVH_score_batch(
            prediction, batch_data, list_DVH_dif=self.list_DVH_dif, dict_DVH_dif=self.dict_DVH_dif, ivs_values=None)

        torch.cuda.empty_cache()

        ckp_re_dir = 'IMAGES_DIRECTORY'+'/linked'
        if batch_idx < 100:
            plot_DVH(prediction, batch_data, path=os.path.join(ckp_re_dir, 'dvh_{}.png'.format(batch_idx)))
            torch.cuda.empty_cache()
            del batch_data
            del prediction
            gc.collect()
        if False:
            prediction_b = prediction.cpu()

            # Post-processing and evaluation
            # prediction[torch.logical_or(possible_dose_mask < 1, prediction < 0)] = 0
            # for save image
            gt_dose[possible_dose_mask < 1] = 0

            pred_img = torch.permute(prediction[0].cpu(), (1, 0, 2, 3))
            gt_img = torch.permute(gt_dose[0].cpu(), (1, 0, 2, 3))
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

        return {"dose_dif": dose_dif}

    def test_epoch_end(self, outputs):

        # ------------- to calculate metrics related to dose distribution predictor --------------
        mean_dose_metric = np.stack([x["dose_dif"] for x in outputs]).mean()
        mean_dvh_metric = np.mean(self.list_DVH_dif)
        for key in self.dict_DVH_dif.keys():
            for metric in self.dict_DVH_dif[key]:
                self.dict_DVH_dif[key][metric] = np.mean(self.dict_DVH_dif[key][metric])

        print(mean_dose_metric, mean_dvh_metric)
        print('----------------------Difference DVH for each structures---------------------')
        print(self.dict_DVH_dif)

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("mean_dvh_metric", mean_dvh_metric)
        return self.dict_DVH_dif


def main(ckpt_file_path_dose, ckpt_file_path_seg, freez=True):
    # initialise the LightningModule
    openkbp = OpenKBPDataModule()
    config_param = {
        "act": 'mish',
        "multiS_conv": True,
        'lr': 3 * 1e-4,
        'weight_decay': 2 * 1e-4,
        'delta1': 10,
        'delta2': 1,
    }
    net = CoupleNet(
        config_param,
        ckpt_file_path_dose,
        ckpt_file_path_seg,
        lr_scheduler_type='cosine',
        eta_min=1e-7,
        last_epoch=-1,
        freez=freez
    )

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_MODEL_DIR_FINAL_COUPLE,
        save_last=True, monitor="mean_dose_score", mode="max",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
        #  filename=net.filename,
    )

    # set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='EXPERIMENT_NAME',
        tracking_uri="databricks",
        # run_name = 'RUN_NAME'
        # run_id = 'RUN_ID'
    )

    # {'act': 'mish', 'multiS_conv': True, 'lr': 0.0002840195762381102, 'weight_decay': 0.00021139244378558662

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
        default_root_dir=config.CHECKPOINT_MODEL_DIR_FINAL_COUPLE,
        # enable_progress_bar=True,
        # log_every_n_steps=net.check_val,
    )

    # train
    # trainer.fit(net, datamodule=openkbp, ckpt_path='SAVED_MODELS'+'/last.ckpt')
    # trainer.fit(net, datamodule=openkbp, ckpt_path='SAVED_MODELS'+'/last.ckpt')
    trainer.fit(net, datamodule=openkbp)

    return net


if __name__ == '__main__':
    net_ = main()
