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
    RandRotate90d

)
import OARSegmentation.config as config

OAR_NAMES = config.OAR_NAMES

OAR_NAMES_DIC = {
    'Brainstem': 1,
    'SpinalCord': 2,
    'RightParotid': 3,
    'LeftParotid': 4,
    'Esophagus': 5,
    'Larynx': 6,
    'Mandible': 7
}


def read_data(dataset_path, indices):
    data_files = []
    list_names = OAR_NAMES
    
    for file in glob(dataset_path):

        patient = {}

        for oar in list_names:

            label_path = glob(file + '/{}.nii.gz'.format(oar))

            if len(label_path):
                label_path = label_path[0]
            else:
                continue

            patient[oar] = label_path

        image_path = glob(file + '/ct.nii.gz')[0]
        patient['CT'] = image_path

        data_files.append(patient)
        
    if indices:
       return [data_files[i] for i in indices]

    return data_files


class Empty2FullOAR:
    def __call__(self, image_dict):
        mask = np.zeros(image_dict["CT"].shape, np.uint8)
        image_dict["OARs"] = mask.copy()
        for oar in OAR_NAMES:
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
        # for oar in self.oar_names:
        #     if oar in image_dict.keys():
        #         mask = np.int8(np.logical_or(image_dict[oar], mask))
        #
        # image_dict["OARs"] = mask

        # image_dict["OARs"] = np.concatenate([image_dict[OAR] for OAR in self.oar_names], axis=0)

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
        image_dict['CT'] = ct.astype(np.float32) / 1000
        return image_dict


class NoneTransform(object):
    def __call__(self, image_dict):
        return image_dict


def prepare_data(files, state, a_min, a_max, cache_num, cache):
    keys = OAR_NAMES + ['CT', 'OARs']
    final_keys = ['CT', 'OARs']

    transforms = Compose(

        [
            LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
            Empty2FullOAR(),
            ORTransform(),
            
            # EnsureChannelFirstd(keys=final_keys, allow_missing_keys=True),  # for version upper than 0.7.0 monai
            AddChanneld(keys=final_keys, allow_missing_keys=True), # for version = 0.7.0 monai
            
            MyIntensityNormalTransform(a_min=a_min, a_max=a_max),
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

            DeleteItemsd(keys=OAR_NAMES),
            ###################################################
            # training transform for data augmentation purposes
            RandCropByPosNegLabeld(
                keys=final_keys,
                label_key="OARs",
                spatial_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE),
                pos=2,
                neg=1,
                num_samples=4,
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
            # monai.transforms.RandFlipd(),
            # monai.transforms.RandRotated(),
            # monai.transforms.Affined(),
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


def get_dataset(path, size, state="train", a_min=-1024, a_max=1500, cache_num=24, cache=False, indices=None):
    # train_path = 'nifti-train-pats/pt_*'
    # test_path = 'nifti-test-pats/pt_*'

    # read files to make dictionary
    files = read_data(path, indices)

    if len(files) == 0:
        raise Exception("number of files are zero!")

    data = prepare_data(files=files[0:size], state=state,
                        a_min=a_min, a_max=a_max, cache_num=cache_num, cache=cache)

    return data


def test():
    # provide OpenKBP dataset
    train_path = 'D:/python_code/thesis_final/provided-data/nifti-train-pats/pt_*'  # 200
    test_path = 'D:/python_code/thesis_final/provided-data/nifti-test-pats/pt_*'  # 100
    train_data = get_dataset(path=config.TRAIN_DIR, state='train', size=config.TRAIN_SIZE,
                             cache_num=24, cache=True)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True,
                              collate_fn=list_data_collate, )

    val_data = get_dataset(path=config.VAL_DIR, state='val', size=config.VAL_SIZE,
                           cache_num=6, cache=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    # sample = next(iter(train_loader))
    # volume = sample['CT']
    # label = sample['OARs']
    # print(volume.shape)
    # print(label.shape)

    for batch_idx, batch_data in enumerate(train_loader):
        volume = batch_data['CT']
        label = batch_data['OARs']
        print(volume.shape)
        print(label.shape)
        print()


if __name__ == '__main__':
    test()