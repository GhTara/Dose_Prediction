import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import multiprocessing

import sys

sys.path.insert(0, 'HOME_DIRECTORY')

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
IMAGE_SIZE = 96
CHANNEL_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 1300
LOAD_MODEL = False
SAVE_MODEL = True
PRETRAIN = False
SW_BATCH_SIZE = 4
MAIN_PATH = 'HOME_DIRECTORY'
CHECKPOINT_MODEL_DIR_PROVIDED_SEG_FTUNE = "YourModels/OARSegmentation/ms_unetr_300/"
CHECKPOINT_MODEL_DIR_PRIVATE_SEG = "YourModels/OARSegmentation/ms_unetr_96/"
CHECKPOINT_MODEL_DIR_PRIVATE_SEG_FTUNE = "YourModels/OARSegmentation/ms_unetr_ftune/"
CHECKPOINT_RESULT_DIR = 'YourSampleImages/OARSegmentation/ms_unetr_300/images/'

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

# for monai == 0.7.0
post_label = AsDiscrete(to_onehot=True, n_classes=len(OAR_NAMES)+1)
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=len(OAR_NAMES)+1)
