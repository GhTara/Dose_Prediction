# -*- encoding: utf-8 -*-
import os


import argparse
# try on colab
import sys
sys.path.insert(0, '/content/drive/.shortcut-targets-by-id/1G1XahkS3Mp6ChD2Q5kBTmR9Cb6B7JUPy/thesis/')
print(sys.path)
from RTDosePrediction.Src.DataLoader.dataloader_OpenKBP_C3D_monai import get_dataset
from RTDosePrediction.Src.NetworkTrainer.network_trainer import NetworkTrainer
from model import Model
from online_evaluation import online_evaluation
from loss import Loss
import RTDosePrediction.Src.DataLoader.config as config
from torch.utils.data import DataLoader
from monai.data import DataLoader, list_data_collate, decollate_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training (default: 1)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[0],
                        help='list_GPU_ids for training (default: [0])')
    parser.add_argument('--max_iter',  type=int, default=80000,
                        help='training iterations(default: 80000)')
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'C3D'
    # In order to record log
    trainer.setting.output_dir = os.path.join(config.OUT_DIR , 'C3D')
    list_GPU_ids = args.list_GPU_ids

    trainer.setting.network = Model(in_ch=9, out_ch=1,
                                    list_ch_A=[-1, 16, 32, 64, 128, 256],
                                    list_ch_B=[-1, 32, 64, 128, 256, 512])

    trainer.setting.max_iter = args.max_iter

    # trainer.setting.train_loader, trainer.setting.val_loader = get_loader(
    #     train_bs=args.batch_size,
    #     val_bs=1,
    #     train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch
    #     val_num_samples_per_epoch=1,
    #     num_works=2
    # )
    
    train_data = get_dataset(path=config.MAIN_PATH + config.TRAIN_DIR, state='train',
                             size=24, cache=True)

    val_data = get_dataset(path=config.MAIN_PATH + config.VAL_DIR, state='val',
                           size=6, cache=True)

    dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True,
                            num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True)

    val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS, pin_memory=True)
                                
    trainer.setting.train_loader, trainer.setting.val_loader = dataloader, val_dataloader

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr': 3e-4,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             args={
                                 'T_max': args.max_iter,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    trainer.set_GPU_device(list_GPU_ids)
    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
