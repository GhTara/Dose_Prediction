import os
import sys
import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import multiprocessing


Device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
TRAIN_SIZE = 200
VAL_SIZE = 100
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
SW_BATCH_SIZE = 1
NUM_WORKERS = multiprocessing.cpu_count()
CACHE_RATE = 1.0
LAMBDA_VOXEL = 100
# 128 ____________________________
IMAGE_SIZE = 128
CHANNEL_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
PRETRAIN = False
TRAIN_VAL_DIR = os.path.normpath('provided-data/nifti-val-pats/pt_*')
CHECKPOINT_MODEL_DIR = "YourModels/DosePrediction/ablation1_dose/"
CHECKPOINT_MODEL_DIR_DOSE_SHARED = "YourModels/DosePrediction/dose_shared/"
CHECKPOINT_MODEL_DIR_DOSE_GAN = "YourModels/DosePrediction/dose_gan/"
CHECKPOINT_MODEL_DIR_BASE = "YourModels/DosePrediction/base_dose_shared/"
CHECKPOINT_MODEL_DIR_BASE_FINAL = "YourModels/DosePrediction/final_baseline/"
CHECKPOINT_MODEL_DIR_DOSE_SHARED_SIMPLE = "YourModels/DosePrediction/simple_dose_shared/"
CHECKPOINT_MODEL_DIR_FINAL = "YourModels/DosePrediction/final/"
CHECKPOINT_MODEL_DIR_FINAL_32 = "YourModels/DosePrediction/final32/"
CHECKPOINT_MODEL_DIR_FINAL_FTUNE = "YourModels/DosePrediction/final_refine/"
CHECKPOINT_MODEL_DIR_FINAL_RAY = "YourModels/DosePrediction/final_ray/"
CHECKPOINT_MODEL_DIR_FINAL_KFOLD = "YourModels/DosePrediction/final_kfold/"
CHECKPOINT_MODEL_DIR_FINAL_LINKED = "YourModels/DosePrediction/linked/"

CHECKPOINT_RESULT_DIR = "YourSampleImages/DosePrediction/vitgen_multiS_dec_random_crop_300/"

TRAIN_DIR = 'provided-data/nifti-train-pats/pt_*'
VAL_DIR = 'provided-data/nifti-test-pats/pt_*'
MAIN_PATH = 'HOME_DIRECTORY'

OAR_NAMES = [
    'Brainstem',
    'SpinalCord',
    'RightParotid',
    'LeftParotid',
    'Esophagus',
    'Larynx',
    'Mandible'
]
PTV_NAMES = ['PTV70',
             'PTV63',
             'PTV56']

# for monai > 0.7.0
post_label = AsDiscrete(to_onehot=len(OAR_NAMES) + 1)
post_pred = AsDiscrete(argmax=True, to_onehot=len(OAR_NAMES) + 1)
