# from pytorch_lightning.loggers import TensorBoardLogger
# from ray import tune, air
# from ray.air import session
# from ray.tune import CLIReporter
# from ray.tune.integration.mlflow import mlflow_mixin
# from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.suggest.ax import AxSearch
from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray_lightning import RayStrategy
from ray import tune

from DosePrediction.Train.train_light_final import *
import DosePrediction.Train.config as config

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def _del_model(self, *_):
        pass

    def _save_model(self, *_):
        pass


def train_model(config_hparam,
                num_epochs=10,
                num_workers=1,
                use_gpu=True,
                callbacks=None):
    model = CascadeUNet(config_hparam,
                        lr_scheduler_type='cosine',
                        eta_min=1e-7,
                        last_epoch=-1
                        )

    callbacks = callbacks or [MyModelCheckpoint(monitor='val_loss')]

    openkbp = OpenKBPDataModule()

    # set up logger
    mlflow_logger = MLFlowLogger(
        experiment_name='EXPERIMENT_NAME',
        tracking_uri="databricks",
        # run_id = ''
    )

    trainer = pl.Trainer(
        # devices=[0],
        # accelerator="gpu",
        max_epochs=num_epochs,
        callbacks=callbacks,
        checkpoint_callback=False,
        strategy=RayStrategy(num_workers=num_workers, use_gpu=use_gpu),
        # progress_bar_refresh_rate=0,
        logger=mlflow_logger,
        check_val_every_n_epoch=3,
        default_root_dir=config.CHECKPOINT_MODEL_DIR,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=openkbp)


def main(num_samples=10,
         num_epochs=10,
         num_workers=1,
         grace_period=1,
         reduction_factor=2,
         brackets=1,
         use_gpu=True):
    config_hparam = {
        "act": 'mish',
        "multiS_conv": True,
        # 'lr': 0.0002840195762381102, 
        # 'weight_decay': 0.00021139244378558662,
        'delta1': 10,
        'delta2': 8,

        # "act": tune.choice(['mish', 'relu']),
        # "multiS_conv": tune.choice([True, False]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-3),
        # 'delta2': tune.randint(lower=0, upper=10),

        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "use_gpu": use_gpu,
    }

    metrics = {"loss": "val_loss"}
    callbacks = [TuneReportCallback(metrics, on="validation_end"),
                 # # set up checkpoints
                 # ModelCheckpoint(dirpath=config.CHECKPOINT_MODEL_DIR,
                 #     save_last=True, monitor="mean_dose_score", mode="max",
                 #     every_n_epochs=net.check_val,
                 #     auto_insert_metric_name=True,
                 #     #  filename=net.filename,
                 # )
                 ]

    trainable = tune.with_parameters(
        train_model,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=callbacks)
    # for continous values
    # searcher = BayesOptSearch(metric="loss", mode="min")
    # for discrete values
    searcher = HyperOptSearch(metric="loss", mode="min")
    # searcher = ConcurrencyLimiter(searcher, max_concurrent=2)
    scheduler = AsyncHyperBandScheduler(
        max_t=num_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        brackets=brackets)

    # resources_per_trial = {"cpu": 2, "gpu": 1}
    resources_per_trial = get_tune_resources(
        num_workers=num_workers, use_gpu=use_gpu)
    analysis = tune.run(
        trainable,
        metric="loss",
        mode="min",
        config=config_hparam,
        num_samples=num_samples,
        resources_per_trial=resources_per_trial,
        scheduler=scheduler,
        # search_alg=searcher,
        name="ASHA/BO",
        verbose=True)

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=20,
         num_epochs=50,
         num_workers=1,
         grace_period=3,
         reduction_factor=2,
         brackets=1,
         use_gpu=True)
