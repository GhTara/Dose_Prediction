import sys
sys.path.insert(0, '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/')
from abc import ABC, abstractmethod

from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.apps import CrossValidation

import pytorch_lightning as pl

import gc

import bitsandbytes as bnb

from DosePrediction.Train.my_model import *
# from RTDosePrediction.DosePrediction.Train.model import *
from DosePrediction.DataLoader.dataloader_OpenKBP_C3D_monai import get_dataset
import DosePrediction.Train.config as config
from DosePrediction.Evaluate.evaluate_openKBP import *
from DosePrediction.Train.loss import GenLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """

    def __init__(
        self,
        data,
        transform,
        cache_num=24, 
        cache_rate=config.CACHE_RATE, 
        num_workers=config.NUM_WORKERS
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
        

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
                                     size=40, cache=True)
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

        self.max_epochs = 200
        # self.max_epochs = 150
        # 10
        self.check_val = 5
        # 5
        self.warmup_epochs = 2

        self.img_height = config.IMAGE_SIZE
        self.img_width = config.IMAGE_SIZE
        self.img_depth = config.IMAGE_SIZE
        
        self.sw_batch_size = config.SW_BATCH_SIZE
        
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
        optimizer = bnb.optim.Adam8bit(self.model_.parameters(), lr=self.lr_scheduler_type, weight_decay=self.weight_decay)
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
        
        dose_dif, DVH_dif, self.dict_DVH_dif, ivs_values = get_Dose_score_and_DVH_score_batch(
            prediction, batch_data, list_DVH_dif=self.list_DVH_dif, dict_DVH_dif=self.dict_DVH_dif, ivs_values=self.ivs_values)
        self.list_DVH_dif.append(DVH_dif)
        self.ivs_values = ivs_values
        
        torch.cuda.empty_cache()
        ckp_re_dir = '/content/drive/MyDrive/results_thesis_images/proposed'
        if False:
        # if batch_idx<12:
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

        self.log("mean_dose_metric", mean_dose_metric)
        self.log("std_dose_metric", std_dose_metric)
        self.log("mean_dvh_metric", mean_dvh_metric)
        self.log("std_dvh_metric", std_dvh_metric)
        # self.logger.log_metrics({"mean_dose_score": mean_dose_score}, self.current_epoch + 1)
        # self.logger.log_metrics({"val_loss": avg_loss}, self.current_epoch + 1)
        return self.dict_DVH_dif


def main(freez=True, start_fold=0, ckpt=None, id_db=None, is_test=False):
    if is_test:
        test_loader = TestOpenKBPDataModule()
        # Initialise the LightningModule
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
        trainer = pl.Trainer(accelerator='gpu', devices=1)
        trainer.test(net, ckpt_path=ckpt,datamodule=test_loader)   
    else:
        # Load the MNIST dataset
        train_transforms, train_set = get_dataset(path=config.MAIN_PATH + config.TRAIN_DIR, state='train', cv=True,
                                                  size=200, cache=True)
        val_transforms, val_set = get_dataset(path=config.MAIN_PATH + config.TRAIN_VAL_DIR, state='val', cv=True,
                                              size=40, cache=True)
        # Define the number of folds for cross-validation
        # start_fold = 0
        num_fold = 6
        folds =  list(range(0, num_fold))
        
        cvdataset = CrossValidation(
            dataset_cls=CVDataset,
            data=train_set+val_set,
            nfolds=num_fold,
            seed=123456,
            transform=train_transforms,
        )
        
    
        # test_ds = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='val',
        #                                 size=100, cache=True)
        # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
        #                       num_workers=config.NUM_WORKERS, pin_memory=True)
    
    
    

        for fold in list(range(start_fold, num_fold)):
            
            # Get the data loaders for this fold
            print(folds[0:fold] + folds[(fold+1):], '--------------------------------------------------')
            # train_ds = cvdataset.get_dataset(folds=folds[0:fold] + folds[(fold+1):])
            # train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
            #                         num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True)
            
            val_ds = cvdataset.get_dataset(folds=fold, transform=val_transforms)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                    num_workers=config.NUM_WORKERS, pin_memory=True)
    
            # Initialise the LightningModule
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
    
            # # set up checkpoints
            # checkpoint_callback = ModelCheckpoint(
            #     dirpath=config.CHECKPOINT_MODEL_DIR_FINAL_KFOLD+"checkpoints_dose/fold-{}".format(fold),
            #     save_last=True, monitor="mean_dose_score", mode="max",
            #     every_n_epochs=net.check_val,
            #     auto_insert_metric_name=True,
            #     #  filename=net.filename,
            # )
        
            # # set up logger
            # if fold == start_fold and id_db!=None:
            #     mlflow_logger = MLFlowLogger(
            #         experiment_name='/Users/gheshlaghitara@gmail.com/dose_prediction_vitGen',
            #         tracking_uri="databricks",
            #         run_id = id_db
            #         )
            # else:
            #     mlflow_logger = MLFlowLogger(
            #         experiment_name='/Users/gheshlaghitara@gmail.com/dose_prediction_vitGen',
            #         tracking_uri="databricks",
            #         run_name = 'exp1_k-fold-{}'.format(fold)
            #         )
    
            # # initialise Lightning's trainer.
            # trainer = pl.Trainer(
            #     # strategy=strategy,
            #     devices=[0],
            #     accelerator="gpu",
            #     max_epochs=net.max_epochs,
            #     check_val_every_n_epoch=net.check_val,
            #     callbacks=[checkpoint_callback],
            #     logger=mlflow_logger,
            #     # callbacks=RichProgressBar(),
            #     # callbacks=[bar],
            #     default_root_dir=config.CHECKPOINT_MODEL_DIR_FINAL_KFOLD,
            #     # enable_progress_bar=True,
            #     # log_every_n_steps=net.check_val,
            # )
            trainer = pl.Trainer(accelerator='gpu', devices=1)
            
            trainer.test(net, ckpt_path=ckpt,dataloaders=val_loader)
            
            # if fold == start_fold and ckpt!=None:
            #     # Load the last checkpoint in this fold and resume training for additional epochs
            #     trainer.fit(net, train_loader, val_loader,
            #             ckpt_path=ckpt)
            # else:
            #     # Train the model for multiple epochs and save the checkpoint after each epoch in this fold
            #     trainer.fit(net, train_loader, val_loader)
            
            # release GPU memory
            torch.cuda.empty_cache()
            
            # release system memory
            gc.collect()
            break
        

    return net


if __name__ == '__main__':
    net_ = main()