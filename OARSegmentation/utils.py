import torch
import os
from monai.losses import DiceLoss
from monai.utils import first
import matplotlib.pyplot as plt
import config
from monai.data import decollate_batch
from monai.metrics import DiceMetric


def get_metric(predicted, target):
    dice_value = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value


# define metric function to match MONAI API
# def get_metric(y_pred, y):
#     y_pred = [config.post_trans(i) for i in decollate_batch(y_pred)]
#     dice_metric = DiceMetric(include_background=True, reduction="mean")
#     dice_metric(y_pred=y_pred, y=y)
#     metric = dice_metric.aggregate().item()
#     dice_metric.reset()
#     return metric


def save_checkpoint(model, optimizer, epoch, model_dir='results'):
    print("==> saving checkpoint")

    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), }
    torch.save(state, os.path.join(model_dir, "best_metric_model.pth.tar"))


def load_checkpoint(model, optimizer, model_dir='results'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    filename = os.path.join(model_dir, "best_metric_model.pth.tar")
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        print(start_epoch)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, int(start_epoch)


def show_patient(data, slice_number, train=True, test=False):
    check_patient_train, check_patient_test = data

    view_patient_train, view_patient_test = first(check_patient_train), first(check_patient_test)

    if train:
        plt.figure("visualization train", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f'image {slice_number}')
        plt.imshow(view_patient_train['image'][0, 0, :, :, slice_number], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f'label {slice_number}')
        plt.imshow(view_patient_train['label'][0, 0, :, :, slice_number])

        plt.show()

    if test:
        plt.figure("visualization test", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f'image {slice_number}')
        plt.imshow(view_patient_test['image'][0, 0, :, :, slice_number], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f'label {slice_number}')
        plt.imshow(view_patient_test['label'][0, 0, :, :, slice_number])

        plt.show()
