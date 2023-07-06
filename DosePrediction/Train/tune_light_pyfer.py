import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.pb2 import PB2
from ray.tune.search.optuna import OptunaSearch

from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray_lightning import RayStrategy
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch

from DosePrediction.Train.train_light_pyfer import *
import DosePrediction.Train.config as config


class TunePyfer(tune.Trainable):
    """Train a Pytorch Model."""

    def setup(self, config_hparam):
        self.config_hparam = config_hparam
        self.data = OpenKBPDataModule()
        self.model = Pyfer(self.config_hparam, )

        metrics = {"loss": "val_loss"}
        callbacks = [TuneReportCallback(metrics, on="validation_end"), ]
        callbacks = callbacks or []

        # set up logger
        mlflow_logger = MLFlowLogger(
            experiment_name='EXPERIMENT_NAME',
            tracking_uri="databricks",
            # run_id = ''
        )

        self.trainer = pl.Trainer(
            max_epochs=config_hparam["num_epochs"],
            strategy=RayStrategy(num_workers=config_hparam["num_workers"], use_gpu=config_hparam["use_gpu"]),
            default_root_dir=config.CHECKPOINT_MODEL_DIR,
            logger=mlflow_logger,
            enable_progress_bar=False,
        )
        self.val_loss = float("inf")

    def step(self):
        self.trainer.fit(self.model, datamodule=self.data)
        self.val_loss = self.trainer.callback_metrics["val_loss"]
        return {"loss": self.val_loss}

    def reset_config(self, new_config):
        self.config_hparam = new_config
        self.model.update_config(new_config)
        return True


class RayTuner:
    def __init__(self, config_hparam):
        self.config_hparam = config_hparam
        self.num_epochs = config_hparam["num_epochs"]
        self.num_workers = config_hparam["num_workers"]
        self.use_gpu = config_hparam["use_gpu"]
        self.analysis = None
        ray.init()

    def ASHAScheduler(self, n_samples, grace_period=10, reduction_factor=3, search='RS'):
        if search == 'BO':
            searcher = BayesOptSearch(metric="loss", mode="min")
            self.analysis = tune.run(TunePyfer,
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
            self.analysis = tune.run(TunePyfer,
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
            self.analysis = tune.run(TunePyfer,
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
        self.analysis = tune.run(TunePyfer,
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

    # For final model
    config_hparam = {
        "act": tune.choice(['mish', 'relu']),
        "multiS_conv": tune.choice([True, False]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-3),
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "use_gpu": use_gpu,
    }

    ray_tuners = RayTuner(config_hparam)
    analysis = ray_tuners.ASHAScheduler(n_samples=num_samples, grace_period=1, reduction_factor=2, search='optuna')
    ray_tuners.get_best_param()


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=15, num_epochs=50, num_workers=1, use_gpu=True)
