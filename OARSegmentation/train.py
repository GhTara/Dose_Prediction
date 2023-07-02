import os
import sys

sys.path.insert(0, 'HOME_DIRECTORY')

import torch

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNETR

import OARSegmentation.config as config
from OARSegmentation.DataLoader.provided_dataset import get_dataset
# from private_dataset import get_dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from pytorch_lightning.callbacks import TQDMProgressBar, ProgressBarBase
from pytorch_lightning.loggers import MLFlowLogger

from typing import Optional

from OARSegmentation.OldModels.Networks.modified_unetr import ModifiedUNETR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


class TestOpenKBPDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # Assign val datasets for use in dataloaders

        self.test_data = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='val',
                                     size=4, cache=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)


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
                          num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)


class PrivateDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.val_data, self.train_data = get_dataset(path=config.MAIN_PATH + config.DIR_PRIVATE, cache=False)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True)
        # return DataLoader(self.train_data, batch_size=config.BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True)
        # return DataLoader(self.val_data, batch_size=config.BATCH_SIZE)


class litAutoSeg(pl.LightningModule):
    def __init__(self, pretrain=False, mode_model=0):
        super().__init__()
        self.val_data = None
        self.train_data = None
        self.filename = "last"

        if mode_model == 0:
            self._model = UNETR(
                in_channels=1,
                out_channels=len(config.OAR_NAMES) + 1,
                img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
                # 16 => 4
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                conv_block=True,
                dropout_rate=0.0,
            )

        elif mode_model == 1:
            self._model = ModifiedUNETR(
                in_channels=1,
                out_channels=len(config.OAR_NAMES) + 1,
                img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
                # 16 => 4
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                conv_block=True,
                dropout_rate=0.0,
            )

        if pretrain:
            pre_model = torch.load('PretrainedModels/OARSegmentation/epoch=399-step=32000.ckpt')
            model_dict = self._model.state_dict()
            keyss = list(pre_model['state_dict'].keys())
            for k in keyss:
                pre_model['state_dict'][k.replace('_model.', '')] = pre_model['state_dict'].pop(k)

            missing = tuple({k for k in model_dict.keys() if k not in pre_model['state_dict']})
            print(f"missing in pretrained: {len(missing)}")
            inside = tuple({k for k in pre_model['state_dict'] if
                            (k in model_dict.keys()) and
                            (model_dict[k].shape == pre_model['state_dict'][k].shape)
                            })
            print(f"inside pretrained: {len(inside)}")
            unused = tuple({k for k in pre_model['state_dict'] if k not in model_dict.keys()})
            print(f"unused pretrained: {len(unused)}")

            pre_model['state_dict'] = {k: v for k, v in pre_model['state_dict'].items() if (k in model_dict.keys()) and
                                       (model_dict[k].shape == pre_model['state_dict'][k].shape)
                                       }
            self._model.load_state_dict(pre_model['state_dict'], strict=False)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        # mean_channel
        self.dice_metric_test = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.dice_metric_list = None

        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean", get_not_nans=False
        )

        self.hd95_metric_test = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean", get_not_nans=False
        )
        self.hd95_metric_list = None

        self.best_metric = 0
        self.best_metric_epoch = 0
        self.train_epoch_loss = []
        self.metric_values = []

        self.max_epochs = 1300
        # 10, 5
        self.check_val = 5
        # 5, 2
        self.warmup_epochs = 2

        self.sw_batch_size = config.SW_BATCH_SIZE

        self.CT_list = []
        self.pred_list = []
        self.gt_list = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch_data, batch_idx):
        train_volume, train_label = batch_data['CT'], batch_data['OARs']
        outputs = self.forward(train_volume)
        train_loss = self.loss_function(outputs, train_label)
        tensorboard_logs = {"train_loss": train_loss.item()}
        return {"loss": train_loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.train_epoch_loss.append(train_mean_loss.detach().cpu().numpy())
        self.logger.log_metrics({"train_loss": train_mean_loss}, self.current_epoch + 1)

    def validation_step(self, batch_data, batch_idx):
        val_volume = batch_data['CT']
        val_label = batch_data['OARs']
        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        val_outputs = sliding_window_inference(
            val_volume, roi_size, self.sw_batch_size, self.forward
        )
        val_loss = self.loss_function(val_outputs, val_label)
        val_outputs = [config.post_pred(i) for i in decollate_batch(val_outputs)]
        val_label = [config.post_label(i) for i in decollate_batch(val_label)]
        self.dice_metric(y_pred=val_outputs, y=val_label)
        self.hd95_metric(y_pred=val_outputs, y=val_label)
        return {"val_loss": val_loss, "val_number": len(val_outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_metric = self.dice_metric.aggregate()
        mean_val_hd95_metric = self.hd95_metric.aggregate()
        self.dice_metric.reset()

        mean_val_loss = torch.tensor(val_loss / num_items)

        self.log("val_loss", mean_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.log_metrics({"dice_metric": mean_val_metric}, self.current_epoch + 1)
        self.logger.log_metrics({"hd95_metric": mean_val_hd95_metric}, self.current_epoch + 1)
        self.logger.log_metrics({"val_loss": mean_val_loss}, self.current_epoch + 1)

        tensorboard_logs = {"val_metric": mean_val_metric, "val_loss": mean_val_loss}

        if mean_val_metric > self.best_metric:
            self.best_metric = mean_val_metric
            self.best_metric_epoch = self.current_epoch

        self.metric_values.append(mean_val_metric)
        return {"log": tensorboard_logs}

    def test_step(self, batch_data, batch_idx):
        test_volume = batch_data['CT']
        test_label = batch_data['OARs']
        roi_size = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE)
        test_outputs = sliding_window_inference(
            test_volume, roi_size, self.sw_batch_size, self.forward
        )

        test_outputs = [config.post_pred(i) for i in decollate_batch(test_outputs)]
        test_label = [config.post_label(i) for i in decollate_batch(test_label)]

        print(test_volume.shape)
        self.CT_list.append(torch.permute(test_volume[0].cpu(), (3, 0, 1, 2)))
        self.pred_list.append(torch.permute(test_outputs[0].cpu(), (3, 0, 1, 2)))
        self.gt_list.append(torch.permute(test_label[0].cpu(), (3, 0, 1, 2)))

        if not os.path.isdir(config.CHECKPOINT_RESULT_DIR):
            os.mkdir(config.CHECKPOINT_RESULT_DIR)

        # save_image(pred_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'pred_{}.jpg'.format(batch_idx)))
        # save_image(gt_grid, os.path.join(config.CHECKPOINT_RESULT_DIR, 'gt_{}.jpg'.format(batch_idx)))

        # plt.imsave(os.path.join(config.CHECKPOINT_RESULT_DIR, 'pred_{}.jpg'.format(batch_idx)), pred_grid[0, :, :])
        # plt.imsave(os.path.join(config.CHECKPOINT_RESULT_DIR, 'gt_{}.jpg'.format(batch_idx)), gt_grid[0, :, :])

        # test_outputs = [config.post_pred(i) for i in decollate_batch(test_outputs)]
        # test_label = [config.post_label(i) for i in decollate_batch(test_label)]

        # plt.imshow(test_label[0][0, :, :, 112].cpu().numpy())
        # plt.show()
        # print(test_outputs[0].shape, test_label[0].shape)

        # dice_metric = self.dice_metric_test(y_pred=test_outputs, y=test_label)
        # if self.dice_metric_list==None:
        #     self.dice_metric_list = dice_metric
        # else:
        #     self.dice_metric_list = torch.cat((self.dice_metric_list, dice_metric), 0)

        # hd95_metric = self.hd95_metric_test(y_pred=test_outputs, y=test_label)
        # if self.hd95_metric_list==None:
        #     self.hd95_metric_list = hd95_metric
        # else:
        #     self.hd95_metric_list = torch.cat((self.hd95_metric_list, hd95_metric), 0)

    def test_epoch_end(self, outputs):
        # classes = OAR_NAMES_DIC = ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid', 'Esophagus', 'Larynx',
        # 'Mandible'] dice_metric_each_class = self.dice_metric_list.nanmean(dim=0) dice_each_class = dict(zip(
        # classes, np.array(dice_metric_each_class.cpu()))) print(dice_each_class)
        print('__________________________________________')
        # hd95_metric_each_class = self.hd95_metric_list.nanmean(dim=0)
        # hd95_each_class = dict(zip(classes, np.array(hd95_metric_each_class.cpu())))
        # print(hd95_each_class)
        return self.CT_list, self.pred_list, self.gt_list


def main(pretrain, mode_model):
    openkbp = OpenKBPDataModule()
    # private_data = PrivateDataModule()

    # initialise the LightningModule
    net = litAutoSeg(pretrain, mode_model)

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_MODEL_DIR_PROVIDED_SEG_FTUNE,
        # dirpath=config.CHECKPOINT_MODEL_DIR_PRIVATE_SEG_FTUNE,
        save_last=True, monitor="val_loss", mode="min",
        every_n_epochs=net.check_val,
        auto_insert_metric_name=True,
        #  filename=net.filename,
    )

    # set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='EXPERIMENT_NAME',
        tracking_uri="databricks",
        run_id='run_id',
        # run_name='run_name'
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
        default_root_dir=config.CHECKPOINT_MODEL_DIR_PROVIDED_SEG_FTUNE,
        # default_root_dir = config.CHECKPOINT_MODEL_DIR_PRIVATE_SEG_FTUNE,
        # enable_progress_bar=True,
        # log_every_n_steps=net.check_val,
    )

    # train trainer.fit(net,
    # ckpt_path=os.path.join(config.CHECKPOINT_MODEL_DIR_PRIVATE_SEG_FTUNE, 'last.ckpt'),
    # datamodule=private_data)
    trainer.fit(net,
                ckpt_path=os.path.join(config.CHECKPOINT_MODEL_DIR_PROVIDED_SEG_FTUNE, 'last.ckpt'),
                datamodule=openkbp)
    # trainer.fit(net, datamodule=private_data)

    return net


if __name__ == '__main__':
    net = main()
