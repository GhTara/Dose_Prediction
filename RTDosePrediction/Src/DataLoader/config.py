import os
import sys
import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import multiprocessing

# if os.path.abspath('..') not in sys.path:
#     sys.path.insert(0, os.path.abspath('..'))


Device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
TRAIN_SIZE = 200
VAL_SIZE = 100
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
SW_BATCH_SIZE = 1
NUM_WORKERS = multiprocessing.cpu_count()
CACHE_RATE = 1.0
# 96 => 16
LAMBDA_VOXEL = 100
# 128 ____________________________
IMAGE_SIZE = 128
CHANNEL_IMG = 3
L1_LAMBDA = 100
# LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
PRETRAIN = False
MAIN_PATH = os.path.normpath('D:/python_code/thesis_final/')
TRAIN_VAL_DIR = os.path.normpath('provided-data-v2/nifti-val-pats/pt_*')
# MAIN_PATH = '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/'
# MAIN_PATH = '/content/drive/MyDrive/thesis/'
CHECKPOINT_MODEL_DIR = "/content/drive/MyDrive/results_thesis/ablation1_dose/"
CHECKPOINT_MODEL_DIR_DOSE_SHARED = "/content/drive/MyDrive/results_thesis/dose_shared/"
CHECKPOINT_MODEL_DIR_DOSE_GAN = "/content/drive/MyDrive/results_thesis/dose_gan/"
CHECKPOINT_MODEL_DIR_BASE = "/content/drive/MyDrive/results_thesis/base_dose_shared/"
CHECKPOINT_MODEL_DIR_BASE_FINAL = "/content/drive/MyDrive/results_thesis/final_baseline/"
CHECKPOINT_MODEL_DIR_BASE_FINAL_FREEZ = "/content/drive/MyDrive/results_thesis/final_baseline_freez/"
CHECKPOINT_MODEL_DIR_DOSE_SHARED_SIMPLE = "/content/drive/MyDrive/results_thesis/simple_dose_shared/"
CHECKPOINT_MODEL_DIR_FINAL = "/content/drive/MyDrive/results_thesis/final/"
CHECKPOINT_MODEL_DIR_FINAL_32 = "/content/drive/MyDrive/results_thesis/final32/"
CHECKPOINT_MODEL_DIR_FINAL_FTUNE = "/content/drive/MyDrive/results_thesis/final_refine/"
CHECKPOINT_MODEL_DIR_FINAL_RAY = "/content/drive/MyDrive/results_thesis/final_ray/"
CHECKPOINT_MODEL_DIR_FINAL_KFOLD = "/content/drive/MyDrive/results_thesis/final_kfold/"

CHECKPOINT_RESULT_DIR = "/content/drive/MyDrive/results_thesis_images/vitgen_multiS_dec_random_crop_300/"

TRAIN_DIR = 'provided-data-v2/nifti-train-pats/pt_*'
VAL_DIR = 'provided-data-v2/nifti-test-pats/pt_*'
MAIN_PATH = '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/'
# MAIN_PATH = os.path.normpath('/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/')
# TRAIN_DIR = os.path.normpath('/provided-data/nifti-train-pats/pt_*')
# VAL_DIR = os.path.normpath('/provided-data/nifti-test-pats/pt_*')
OUT_DIR = os.path.normpath("/content/drive/MyDrive/results_thesis/output_ge/")

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

# post_trans = Compose(
#         [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
#     )


# post_pred = AsDiscrete(argmax=True, to_onehot=len(OAR_NAMES)+1)
# post_label = AsDiscrete(to_onehot=len(OAR_NAMES)+1)

# # for monai == 0.7.0
# post_label = AsDiscrete(to_onehot=True, n_classes=len(OAR_NAMES) + 1)
# post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=len(OAR_NAMES) + 1)

# for monai > 0.7.0
post_label = AsDiscrete(to_onehot=len(OAR_NAMES) + 1)
post_pred = AsDiscrete(argmax=True, to_onehot=len(OAR_NAMES) + 1)
