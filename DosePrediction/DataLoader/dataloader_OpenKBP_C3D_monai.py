from glob import glob

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

PATH_DATASETS = os.environ.get("PATH_DATASETS", "")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

from monai.data import Dataset, DataLoader, CacheDataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Orientationd,
    ConcatItemsd,
    DeleteItemsd,
    # ToMetaTensord,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    Transposed,

)
import DosePrediction.Train.config as config

OAR_NAMES = config.OAR_NAMES
PTV_NAMES = config.PTV_NAMES

OAR_NAMES_DIC = {
    'Brainstem': 1,
    'SpinalCord': 2,
    'RightParotid': 3,
    'LeftParotid': 4,
    'Esophagus': 5,
    'Larynx': 6,
    'Mandible': 7
}


def read_data(dataset_path):
    data_files = []
    list_names = OAR_NAMES + PTV_NAMES
    patient_files = glob(dataset_path)
    # random.shuffle(patient_files)
    for file in patient_files:

        patient = {}

        for oar in list_names:

            label_path = glob(file + '/{}.nii.gz'.format(oar))

            if len(label_path):
                label_path = label_path[0]
            else:
                continue

            patient[oar] = label_path

        image_path = glob(file + '/CT.nii.gz')[0]
        patient['CT'] = image_path

        image_path = glob(file + '/dose.nii.gz')[0]
        patient['dose'] = image_path

        image_path = glob(file + '/possible_dose_mask.nii.gz')[0]
        patient['dose_mask'] = image_path
        
        patient['file_path'] = image_path

        data_files.append(patient)

        # data_files.shuffle()

    return data_files


class Empty2FullOAR:
    def __call__(self, image_dict):
        mask = np.zeros(image_dict["CT"].shape, np.uint8)
        # image_dict["OARs"] = mask.copy()
        image_dict["PTV"] = mask.copy()
        for oar in OAR_NAMES + PTV_NAMES:
            if oar in image_dict.keys():
                None
            else:
                image_dict[oar] = mask.copy()

        return image_dict


# preprocess on oars: merge all oars in one mask
class ORTransform:
    def __init__(self):
        self.oar_names = OAR_NAMES_DIC

    def __call__(self, image_dict):
        # 1. Merge OAR together in one channel with their own labels
        # for i, oar_name in enumerate(OAR_NAMES_DIC.keys()):
        #     oar = image_dict[oar_name]
        #     image_dict["OARs"][oar > 0] = OAR_NAMES_DIC[oar_name]

        # 2. Each OAR has their own channel
        image_dict["OARs"] = np.concatenate([image_dict[oar_name] for oar_name in OAR_NAMES], axis=0)

        return image_dict


# preprocess on PTVs: merge all ptvs to make a ptv in normalizing way
class NormalizePTVTr:
    def __call__(self, image_dict):
        ptv70 = image_dict['PTV70'] if 'PTV70' in image_dict.keys() else np.zeros(image_dict["CT"].shape, np.uint8)
        ptv63 = image_dict['PTV63'] if 'PTV63' in image_dict.keys() else np.zeros(image_dict["CT"].shape, np.uint8)
        ptv56 = image_dict['PTV56'] if 'PTV56' in image_dict.keys() else np.zeros(image_dict["CT"].shape, np.uint8)

        image_dict['PTV'] = 70.0 / 70. * ptv70 \
                            + 63.0 / 70. * ptv63 \
                            + 56.0 / 70. * ptv56
        return image_dict


# preprocess on Dose: normalize the dose volume based on max value = 70
class NormalizeDoseTr:
    def __call__(self, image_dict):
        image_dict['real_dose'] = image_dict['dose'].copy()
        # 70 Gy to the high-dose planning
        image_dict['dose'] = image_dict['dose'] / 70.0
        return image_dict


# preprocess on CT: put in an interval (intensity) => then normalize it
class MyIntensityNormalTransform:
    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, image_dict):
        ct = np.clip(image_dict['CT'], a_min=self.a_min, a_max=self.a_max)
        image_dict['CT'] = ct.astype(np.float32) / 1000.
        return image_dict


class NoneTransform(object):
    def __call__(self, image_dict):
        return image_dict
        
        
def prepare_data(files, state, cv, a_min, a_max, cache_num, cache, crop_flag, image_size):
    keys = PTV_NAMES + OAR_NAMES + ['CT', 'dose', 'dose_mask']
    # final_keys = ['PTV'] + OAR_NAMES + ['CT', 'dose', 'dose_mask']
    final_keys = ['PTV'] + keys + ['real_dose']
    final_keys2 = final_keys + ['Input', 'GT'] if state == 'test' else ['Input', 'GT']

    transforms = Compose(

        [
            LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
            Empty2FullOAR(),
            # comment for ablation
            Transposed(keys=keys, indices=[2, 1, 0]),
            # ORTransform(),
            NormalizePTVTr(),
            MyIntensityNormalTransform(a_min=a_min, a_max=a_max),
            NormalizeDoseTr(),

            # EnsureChannelFirstd(keys=final_keys),  # for version upper than 0.7.0 monai
            AddChanneld(keys=final_keys, allow_missing_keys=True),  # for version = 0.7.0 monai

            
            
            # CropForegroundd(keys=final_keys, source_key="CT"),
            # # Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),

            # ToMetaTensord(keys=final_keys),  # for version upper than 0.7.0 monai

            Orientationd(keys=final_keys, axcodes='RAS'),
            # # change contrast (to make it visibility) => normalize it between 0 and 1
            # # ScaleIntensityRanged(keys='image', a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            # # crop black part of CT following that, corresponding label
            # CropForegroundd(keys=final_keys, source_key='CT'),
            # ORTransform(),  # OR transform on OARs
            # EnsureChannelFirstd(keys=keys+['OARs'], allow_missing_keys=True),
            # Resized(keys=final_keys, spatial_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
            #         allow_missing_keys=True, mode="nearest"),
            RandShiftIntensityd(
                keys=["CT"],
                offsets=0.10,
                prob=0.50,
            ) if state == "train" else NoneTransform(),

            ConcatItemsd(
                keys=['PTV'] + OAR_NAMES + ['CT'], dim=0, name='Input'
            ),

            ConcatItemsd(
                keys=['dose', 'dose_mask'], dim=0, name='GT'
            ),

            DeleteItemsd(keys=final_keys) if state != 'test' else NoneTransform(),
            ###################################################
            # training transform for data augmentation purposes
            RandCropByPosNegLabeld(
                keys=final_keys2,
                label_key="GT",
                spatial_size=(image_size, image_size, image_size),
                pos=2,
                neg=1,
                num_samples=config.SW_BATCH_SIZE,
                image_key="Input",
                image_threshold=0,
            ) if (state == "train" and crop_flag) else NoneTransform(),
            RandFlipd(
                keys=final_keys2,
                spatial_axis=[0],
                prob=0.10,  # 0.8
            ) if state == "train" else NoneTransform(),
            RandFlipd(  # doesn't exist
                keys=final_keys2,
                spatial_axis=[1],
                prob=0.10,
            ) if state == "train" else NoneTransform(),
            RandFlipd(
                keys=final_keys2,
                spatial_axis=[2],  # 0.8
                prob=0.10,
            ) if state == "train" else NoneTransform(),
            RandRotate90d(  # rotation 40 angels
                keys=final_keys2,
                prob=0.10,
                max_k=3,
            ) if state == "train" else NoneTransform(),
            # random translation

            # monai.transforms.RandFlipd(),
            # monai.transforms.RandRotated(),
            # monai.transforms.Affined(),
            ToTensord(keys=final_keys2)
        ]
    )
    
    if cv:
        return transforms

    if cache:
        # the data exist in gpu memory => epected to be fast
        ds = CacheDataset(data=files, transform=transforms,
                          cache_num=cache_num, cache_rate=config.CACHE_RATE, num_workers=config.NUM_WORKERS)
    else:
        ds = Dataset(data=files, transform=transforms)

    return ds


def get_dataset(path, size, state="train", cv=False, a_min=-1024, a_max=1500, cache_num=24, cache=False, crop_flag=False, image_size=128):
    # train_path = 'nifti-train-pats/pt_*'
    # test_path = 'nifti-test-pats/pt_*'

    # read files to make dictionary
    files = read_data(path)

    if len(files) == 0:
        raise Exception("number of files are zero!")
        
    if cv:
        transforms = prepare_data(files=files[0:size], state=state, cv=cv,
                        a_min=a_min, a_max=a_max, cache_num=cache_num, cache=cache, crop_flag=crop_flag, image_size=image_size)
        return transforms, files[0:size]

    data = prepare_data(files=files[0:size], state=state, cv=cv,
                        a_min=a_min, a_max=a_max, cache_num=cache_num, cache=cache, crop_flag=crop_flag, image_size=image_size)

    return data


def test():
    # provide OpenKBP dataset
    train_path = 'D:/python_code/thesis_final/provided-data/nifti-train-pats/pt_*'  # 200
    test_path = 'D:/python_code/thesis_final/provided-data/nifti-test-pats/pt_*'  # 100
    train_data = get_dataset(path=train_path, state='train', size=1,
                             cache_num=1, cache=True)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True,
                              collate_fn=list_data_collate, )

    val_data = get_dataset(path=test_path, state='val', size=1,
                           cache_num=1, cache=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    # sample = next(iter(train_loader))
    # volume = sample['CT']
    # label = sample['OARs']
    # print(volume.shape)
    # print(label.shape)

    for batch_idx, batch_data in enumerate(train_loader):
        break
    volume = batch_data['CT']
    label = batch_data['OARs']
    ptv = batch_data['PTV']
    dose = batch_data['dose']
    mask_dose = batch_data['dose_mask']
    print(volume.shape)
    print(label.shape)
    print(dose.shape)
    print()
