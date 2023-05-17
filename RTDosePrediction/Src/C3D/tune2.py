import ray
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.schedulers import PopulationBasedTraining, MedianStoppingRule, ASHAScheduler
from ray.tune.schedulers.pb2 import PB2
from ray.tune.search.dragonfly import DragonflySearch
from ray.tune.search.optuna import OptunaSearch

from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray_lightning import RayStrategy
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from RTDosePrediction.Src.C3D.train_light_final import *
import RTDosePrediction.Src.DataLoader.config as config


class TuneCascade(tune.Trainable):
    """Train a Pytorch ConvNet."""

    def setup(self, config_hparam):
        self.config_hparam = config_hparam
        self.data = OpenKBPDataModule()
        self.model = CascadeUNet(self.config_hparam,
                                 lr_scheduler_type='cosine',
                                 eta_min=1e-7,
                                 last_epoch=-1
                                 )
        metrics = {"loss": "val_loss"}
        callbacks = [TuneReportCallback(metrics, on="validation_end"),
                     # ModelCheckpoint(
                     #     dirpath=config.CHECKPOINT_MODEL_DIR,
                     #     save_last=True, monitor="mean_dose_score", mode="max",
                     #     every_n_epochs=self.model.check_val,
                     #     auto_insert_metric_name=True,
                     #     #  filename=net.filename,
                     # )
                     ]
        callbacks = callbacks or []
        
        # set up logger
        mlflow_logger = MLFlowLogger(
            experiment_name='/Users/gheshlaghitara@gmail.com/dose-prediction-tune-hp',
            tracking_uri="databricks",
            # run_id = ''
        )
    
        self.trainer = pl.Trainer(
            max_epochs=config_hparam["num_epochs"],
            # callbacks=callbacks,
            strategy=RayStrategy(num_workers=config_hparam["num_workers"], use_gpu=config_hparam["use_gpu"]),
            # progress_bar_refresh_rate=0,
            default_root_dir=config.CHECKPOINT_MODEL_DIR,
            logger=mlflow_logger,
            enable_progress_bar=False,
        )
        self.val_loss = float("inf")

    def step(self):
        self.trainer.fit(self.model, datamodule=self.data)
        self.val_loss = self.trainer.callback_metrics["val_loss"]
        return {"loss": self.val_loss}

    # def save_checkpoint(self, tmp_ckpt_dir):
    #     path = osp.join(tmp_ckpt_dir, f"{self.runner.config.VARIANT}.{self.runner.count_checkpoints}.pth")
    #     self.runner.save_checkpoint(path)
    #     return path
    #
    # def load_checkpoint(self, path):
    #     self.runner.load_checkpoint(path)

    def reset_config(self, new_config):
        self.config_hparam = new_config
        self.model.update_config(new_config)
        return True


class RayTuner:
    def __init__(self, config_hparam):
        # self.model = model
        # self.data = data
        # self.callbacks = callbacks
        self.config_hparam = config_hparam
        self.num_epochs = config_hparam["num_epochs"]
        self.num_workers = config_hparam["num_workers"]
        self.use_gpu = config_hparam["use_gpu"]
        self.analysis = None
        ray.init()

    # def train_model(self):
    #     data = OpenKBPDataModule()
    #     model = CascadeUNet(self.config_hparam,
    #                         lr_scheduler_type='cosine',
    #                         lr=3e-4,
    #                         weight_decay=1e-4,
    #                         eta_min=1e-7,
    #                         last_epoch=-1
    #                         )
    #     trainer = pl.Trainer(
    #         max_epochs=self.num_epochs,
    #         callbacks=self.callbacks,
    #         strategy=RayStrategy(num_workers=self.num_workers, use_gpu=self.use_gpu),
    #         # progress_bar_refresh_rate=0,
    #         default_root_dir=config.CHECKPOINT_MODEL_DIR,
    #     )
    #
    #     trainer.fit(model, datamodule=data)

    def ASHAScheduler(self, n_samples, grace_period=10, reduction_factor=3, search='RS'):
        if search == 'BO':
            searcher = BayesOptSearch(metric="loss", mode="min")
            self.analysis = tune.run(TuneCascade,
                                     name="ASHA/BO",
                                     verbose=True,
                                     num_samples=n_samples,
                                     resources_per_trial=get_tune_resources(
                                         num_workers=self.num_workers, use_gpu=self.use_gpu),
                                     search_alg=searcher,
                                     scheduler=ASHAScheduler(
                                         time_attr='training_iteration',
                                         metric='loss',
                                         mode='min',
                                         max_t=self.num_epochs,
                                         grace_period=grace_period,
                                         reduction_factor=reduction_factor,
                                         brackets=1),
                                     config=self.config_hparam,
                                     reuse_actors=True)
        elif search == 'optuna ':
            searcher = OptunaSearch(
                        metric="loss",
                        mode="min")
            self.analysis = tune.run(TuneCascade,
                                     name="ASHA/BO2",
                                     verbose=True,
                                     num_samples=n_samples,
                                     resources_per_trial=get_tune_resources(
                                         num_workers=self.num_workers, use_gpu=self.use_gpu),
                                     search_alg=searcher,
                                     scheduler=ASHAScheduler(
                                         time_attr='training_iteration',
                                         metric='loss',
                                         mode='min',
                                         max_t=self.num_epochs,
                                         grace_period=grace_period,
                                         reduction_factor=reduction_factor,
                                         brackets=1),
                                     config=self.config_hparam,
                                     reuse_actors=True)                                     
        else:
            self.analysis = tune.run(TuneCascade,
                                     name="ASHA/RS",
                                     verbose=True,
                                     num_samples=n_samples,
                                     resources_per_trial=get_tune_resources(
                                         num_workers=self.num_workers, use_gpu=self.use_gpu),
                                     scheduler=ASHAScheduler(
                                         time_attr='training_iteration',
                                         metric='loss',
                                         mode='min',
                                         max_t=self.num_epochs,
                                         grace_period=grace_period,
                                         reduction_factor=reduction_factor,
                                         brackets=1),
                                     config=self.config_hparam,
                                     reuse_actors=True,
                                     fail_fast=True)
        return self.analysis

    def PopulationBasedTraining(self, n_samples, perturbation_inter=10, hyperparam_mutations=None):
        pbt = PB2(time_attr='training_iteration',
                  metric='loss',
                  mode='min',
                  perturbation_interval=20.0,
                  hyperparam_bounds=self.config_hparam)
        self.analysis = tune.run(TuneCascade,
                                 name="PB2",
                                 resources_per_trial=get_tune_resources(
                                         num_workers=self.num_workers, use_gpu=self.use_gpu),
                                 scheduler=pbt,
                                 metric="loss",
                                 mode="min",
                                 verbose=True,
                                 stop={
                                     "training_iteration": self.num_epochs,
                                 },
                                 num_samples=n_samples,
                                 fail_fast=True,
                                 config=self.config_hparam,
                                 reuse_actors=True)
        return self.analysis

    def get_best_param(self):
        print("Best config: ", self.analysis.get_best_config(metric="loss"))
        # Get a dataframe for analyzing trial results.
        return self.analysis.dataframe()


def main(num_samples=10,
         num_epochs=10,
         num_workers=1,
         use_gpu=True,
         mode_eval=1):
             
    # # for vit
    # config_hparam = {
    #     "num_layers": tune.choice([4, 8, 12]),
    #     "num_heads": tune.choice([3, 6, 12]),
    #     "num_epochs": num_epochs,
    #     "num_workers": num_workers,
    #     "use_gpu": use_gpu,
    # }
    
    # for final model
    config_hparam = {
        "act": tune.choice(['mish', 'relu']),
        "multiS_conv": tune.choice([True, False]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-3),
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "use_gpu": use_gpu,
    }

    raytuners = RayTuner(config_hparam)
    analysis = raytuners.ASHAScheduler(n_samples=num_samples, grace_period=1, reduction_factor=2, search='optuna')
    raytuners.get_best_param()


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=15, num_epochs=50, num_workers=1, use_gpu=True)
