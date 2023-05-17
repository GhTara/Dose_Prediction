# -*- encoding: utf-8 -*-
import os
import sys
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

import argparse

from DataLoader.dataloader_OpenKBP_C3D import get_loader
from NetworkTrainer.network_trainer import NetworkTrainer
from model import Model
from online_evaluation import online_evaluation
from loss import Loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[1, 0],
                        help='list_GPU_ids for training (default: [1, 0])')
    parser.add_argument('--max_iter',  type=int, default=80000,
                        help='training iterations(default: 80000)')
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'HD-U-Net'
    trainer.setting.output_dir = '../../Output/HD-U-Net'
    list_GPU_ids = args.list_GPU_ids

    trainer.setting.network = Model(in_ch=9, growth_rate=16, upsample_chan=64)

    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader, trainer.setting.val_loader = get_loader(
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch
        val_num_samples_per_epoch=1,
        num_works=4
    )

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
