import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import multiprocessing


Device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
TRAIN_DIR = 'provided-data/nifti-train-pats/pt_*'
VAL_DIR = 'provided-data/nifti-test-pats/pt_*'
DIR_PRIVATE = 'private-data/cropped*'
TRAIN_SIZE = 200
VAL_SIZE = 100
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = multiprocessing.cpu_count()
CACHE_RATE = 1.0
# 96 => 16
IMAGE_SIZE = 96
CHANNEL_IMG = 3
L1_LAMBDA = 100
# LAMBDA_GP = 10
NUM_EPOCHS = 1300
LOAD_MODEL = False
SAVE_MODEL = True
PRETRAIN = False
SW_BATCH_SIZE = 4
MAIN_PATH = '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/'
# MAIN_PATH = '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/'
# MAIN_PATH = '/content/drive/MyDrive/thesis/'
CHECKPOINT_MODEL_DIR = "/content/drive/MyDrive/results_thesis/ms_unetr_300/"
CHECKPOINT_MODEL_DIR_PRIVATE_SEG = "/content/drive/MyDrive/results_thesis/ms_unetr_96/"
CHECKPOINT_MODEL_DIR_PRIVATE_SEG_FTUNE = "/content/drive/MyDrive/results_thesis/ms_unetr_ftune/"
CHECKPOINT_RESULT_DIR = '/content/drive/MyDrive/results_thesis/ms_unetr_300/images/'

OAR_NAMES = [
    'Brainstem',
    'SpinalCord',
    'RightParotid',
    'LeftParotid',
    'Esophagus',
    'Larynx',
    'Mandible'
]

OAR_NAMES_PRIVATE = [
    'BRAIN_STEM',
    'L_EYE',
    'R_EYE',
    'L_LACRIMAL',
    'R_LACRIMAL',
    'L_LENS',
    'R_LENS',
    'L_OPTIC_NERVE',
    'R_OPTIC_NERVE',
    'L_TEMPORAL_LOBE',
    'R_TEMPORAL_LOBE',
    'OPTIC_CHIASM',
    'PITUITARY',
    # 'L_COCHLEA',
    # 'R_COCHLEA',
    # 'L_TMJ',
    # 'R_TMJ',
]

# post_trans = Compose(
#         [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
#     )


# post_pred = AsDiscrete(argmax=True, to_onehot=len(OAR_NAMES)+1)
# post_label = AsDiscrete(to_onehot=len(OAR_NAMES)+1)

# for monai == 0.7.0
post_label = AsDiscrete(to_onehot=True, n_classes=len(OAR_NAMES)+1)
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=len(OAR_NAMES)+1)
