from glob import glob
import numpy as np
import torch

from monai.data import Dataset, DataLoader, CacheDataset, list_data_collate, pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    ConcatItemsd,
    DeleteItemsd,
    Activations,
    # ToMetaTensord,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    SpatialPadd,

)
import config

OAR_NAMES = config.OAR_NAMES_PRIVATE

OAR_NAMES_DIC = {
    'BRAIN_STEM': 1,
    'L_EYE': 2,
    'R_EYE': 3,
    'L_LACRIMAL': 4,
    'R_LACRIMAL': 5,
    'L_LENS': 6,
    'R_LENS': 7,
    'L_OPTIC_NERVE': 8,
    'R_OPTIC_NERVE': 9,
    'L_TEMPORAL_LOBE': 10,
    'R_TEMPORAL_LOBE': 11,
    'OPTIC_CHIASM': 12,
    'PITUITARY': 13,
    # 'L_COCHLEA': 14,
    # 'R_COCHLEA': 15,
    # 'L_TMJ': 16,
    # 'R_TMJ': 17,
    # '': 18,
    # '': 19,
    # '': 20,
}


def read_data(dataset_path):
    data_files = []
    list_names = OAR_NAMES

    for file in glob(dataset_path):

        patient = {}

        patient['filename'] = file

        patient['reverse'] = True


        for oar in list_names:

            label_path = glob(file + '/Segmentation-{}*'.format(oar))

            if len(label_path):
                label_path = label_path[0]
            else:
                continue

            patient[oar] = label_path

        image_path = glob(file + '/CT.nii.gz')[0]
        patient['CT'] = image_path

        data_files.append(patient)

    return data_files


class PermuteD:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, image_dict):
        if image_dict['reverse']:
            for key in self.keys:
                image_dict[key] = image_dict[key][::, ::, ::-1]

        return image_dict


class Empty2FullOAR:
    def __call__(self, image_dict):
        mask = np.zeros(image_dict["CT"].shape, np.uint8)
        image_dict["OARs"] = mask.copy()

        return image_dict


# preprocess on oars: merge all oars in one mask
class ORTransform:

    def __init__(self):
        self.oar_names = OAR_NAMES_DIC

    def __call__(self, image_dict):

        for i, oar_name in enumerate(OAR_NAMES_DIC.keys()):
            oar = image_dict[oar_name]
            image_dict["OARs"][oar > 0] = OAR_NAMES_DIC[oar_name]
            # image_dict["OARs"] = np.add(image_dict["OARs"], oar)

        return image_dict


# preprocess on CT: put in an interval (intensity) => then normalize it
class MyIntensityNormalTransform:
    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, image_dict):
        ct = np.clip(image_dict['CT'], a_min=self.a_min, a_max=self.a_max)
        image_dict['CT'] = ct.astype(np.float32) / 2000
        return image_dict


class NoneTransform(object):
    def __call__(self, image_dict):
        return image_dict


def prepare_data(files, state, a_min, a_max, cache_num, cache):
    keys = OAR_NAMES + ['CT']
    final_keys = ['CT', 'OARs']

    transforms = Compose(

        [
            LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
            # PermuteD(keys=keys),
            Empty2FullOAR(),
            ORTransform(),

            # EnsureChannelFirstd(keys=final_keys, allow_missing_keys=True),  # for version upper than 0.7.0 monai
            AddChanneld(keys=final_keys, allow_missing_keys=True),  # for version = 0.7.0 monai

            Resized(keys=final_keys, spatial_size=(128, 128, -1), mode=['area', 'nearest']),

            MyIntensityNormalTransform(a_min=a_min, a_max=a_max),
            # # CropForegroundd(keys=final_keys, source_key="CT"),
            # # # Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),
            #
            # ToMetaTensord(keys=final_keys),  # for version upper than 0.7.0 monai
            #
            Orientationd(keys=final_keys, axcodes='RAS'),
            # # # change contrast (to make it visibility) => normalize it between 0 and 1
            # # # ScaleIntensityRanged(keys='image', a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            # # # crop black part of CT following that, corresponding label
            # # CropForegroundd(keys=final_keys, source_key='CT'),
            # # ORTransform(),  # OR transform on OARs
            # # EnsureChannelFirstd(keys=keys+['OARs'], allow_missing_keys=True),
            DeleteItemsd(keys=OAR_NAMES),
            SpatialPadd(keys=final_keys, spatial_size=[-1, -1, 128]),
            ###################################################
            # training transform for data augmentation purposes
            RandCropByPosNegLabeld(
                keys=final_keys,
                label_key="OARs",
                spatial_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
                pos=2,
                neg=1,
                num_samples=config.SW_BATCH_SIZE,
                image_key="CT",
                image_threshold=0,
            ) if state == "train" else NoneTransform(),
            RandFlipd(
                keys=final_keys,
                spatial_axis=[0],
                prob=0.10,
            ) if state == "train" else NoneTransform(),
            RandFlipd(
                keys=final_keys,
                spatial_axis=[1],
                prob=0.10,
            ) if state == "train" else NoneTransform(),
            RandFlipd(
                keys=final_keys,
                spatial_axis=[2],
                prob=0.10,
            ) if state == "train" else NoneTransform(),
            RandRotate90d(
                keys=final_keys,
                prob=0.10,
                max_k=3,
            ) if state == "train" else NoneTransform(),
            RandShiftIntensityd(
                keys=["CT"],
                offsets=0.10,
                prob=0.50,
            ) if state == "train" else NoneTransform(),

            ToTensord(keys=final_keys)
        ]
    )

    if cache:
        # the data exist in gpu memory => epected to be fast
        ds = CacheDataset(data=files, transform=transforms,
                          cache_num=cache_num, cache_rate=config.CACHE_RATE, num_workers=config.NUM_WORKERS)
    else:
        ds = Dataset(data=files, transform=transforms)

    return ds


def get_dataset(path, a_min=-2048, a_max=2500, cache_num=24, cache=False):
    # read files to make dictionary
    random_test = [44, 23, 6, 16, 43, 42, 90, 21, 54, 46, 39, 75, 62, 84, 65, 30]  # 16 val-set
    files = read_data(path)

    if len(files) == 0:
        raise Exception("number of files are zero!")

    test_files = [files[i] for i in range(len(files)) if (i in random_test)]
    train_files = [files[i] for i in range(len(files)) if not (i in random_test)]

    test_data = prepare_data(files=test_files, state="val",
                             a_min=a_min, a_max=a_max, cache_num=cache_num, cache=cache)

    train_data = prepare_data(files=train_files, state="train",
                              a_min=a_min, a_max=a_max, cache_num=cache_num, cache=cache)

    return test_data, train_data


def test():
    data = get_dataset(path=config.TRAIN_DIR_PRIVATE, state='train', size=1)

    loader = DataLoader(data, batch_size=config.BATCH_SIZE)

    for batch_idx, batch_data in enumerate(loader):
        volume = batch_data['CT']
        # label = batch_data['OARs']


if __name__ == '__main__':
    test()
