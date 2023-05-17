# -*- encoding: utf-8 -*-
import time

import torch
import torch.nn as nn
from torch import optim
'''
Used for training the baseline according its repository: 
https://github.com/LSL000UD/RTDosePrediction 
'''
class TrainerSetting:
    def __init__(self):
        self.project_name = None
        # Path for saving model and training log
        self.output_dir = None

        # Generally only use one of them
        self.max_iter = 99999999
        self.max_epoch = 99999999

        # Default not use this,
        # because the models of "best_train_loss", "best_val_evaluation_index", "latest" have been saved.
        self.save_per_epoch = 99999999
        self.eps_train_loss = 0.01

        self.network = None
        self.device = None
        self.list_GPU_ids = None

        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_type = None

        # Default update learning rate after each epoch
        self.lr_scheduler_update_on_iter = False

        self.loss_function = None

        # If do online evaluation during validation
        self.online_evaluation_function_val = None


class TrainerLog:
    def __init__(self):
        self.iter = -1
        self.epoch = -1

        # Moving average loss, loss is the smaller the better
        self.moving_train_loss = None
        # Average train loss of a epoch
        self.average_train_loss = 99999999.
        self.best_average_train_loss = 99999999.
        # Evaluation index is the higher the better
        self.average_val_index = -99999999.
        self.best_average_val_index = -99999999.

        # Record changes in training loss
        self.list_average_train_loss_associate_iter = []
        # Record changes in validation evaluation index
        self.list_average_val_index_associate_iter = []
        # Record changes in learning rate
        self.list_lr_associate_iter = []

        # Save status of the trainer, eg. best_train_loss, latest, best_val_evaluation_index
        self.save_status = []


class TrainerTime:
    def __init__(self):
        self.train_time_per_epoch = 0.
        # Time for loading data, eg. data precessing, data augmentation and moving tensors from cpu to gpu
        # In fact, most of the time is spent on moving tensors from cpu to gpu, something like doing multi-processing on
        # CUDA tensors cannot succeed in Windows,
        # you may use cuda.Steam to accelerate it. https://github.com/NVIDIA/apex, but it needs larger GPU memory
        self.train_loader_time_per_epoch = 0.

        self.val_time_per_epoch = 0.
        self.val_loader_time_per_epoch = 0.


class NetworkTrainer:
    def __init__(self):
        self.log = TrainerLog()
        self.setting = TrainerSetting()
        self.time = TrainerTime()

    def set_GPU_device(self, list_GPU_ids):
        self.setting.list_GPU_ids = list_GPU_ids
        sum_device = len(list_GPU_ids)
        # cpu only
        if list_GPU_ids[0] == -1:
            self.setting.device = torch.device('cpu')
        # single GPU
        elif sum_device == 1:
            self.setting.device = torch.device('cuda')
        # multi-GPU
        else:
            self.setting.device = torch.device('cuda:' + str(list_GPU_ids[0]))
            self.setting.network = nn.DataParallel(self.setting.network, device_ids=list_GPU_ids)
        self.setting.network.to(self.setting.device)

    def set_optimizer(self, optimizer_type, args):
        # Sometimes we need set different learning rates for "encoder" and "decoder" separately
        if optimizer_type == 'Adam':
            if hasattr(self.setting.network, 'decoder') and hasattr(self.setting.network, 'encoder'):
                self.setting.optimizer = optim.Adam([
                    {'params': self.setting.network.encoder.parameters(), 'lr': args['lr_encoder']},
                    {'params': self.setting.network.decoder.parameters(), 'lr': args['lr_decoder']}
                ],
                    weight_decay=args['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    amsgrad=True)
            else:
                self.setting.optimizer = optim.Adam(self.setting.network.parameters(),
                                                    lr=args['lr'],
                                                    weight_decay=3e-5,
                                                    betas=(0.9, 0.999),
                                                    eps=1e-08,
                                                    amsgrad=True)

    def set_lr_scheduler(self, lr_scheduler_type, args):
        if lr_scheduler_type == 'step':
            self.setting.lr_scheduler_type = 'step'
            self.setting.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.setting.optimizer,
                                                                       milestones=args['milestones'],
                                                                       gamma=args['gamma'],
                                                                       last_epoch=args['last_epoch']
                                                                       )
        elif lr_scheduler_type == 'cosine':
            self.setting.lr_scheduler_type = 'cosine'
            self.setting.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.setting.optimizer,
                                                                             T_max=args['T_max'],
                                                                             eta_min=args['eta_min'],
                                                                             last_epoch=args['last_epoch']
                                                                             )
        elif lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler_type = 'ReduceLROnPlateau'
            self.setting.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.setting.optimizer,
                                                                             mode='min',
                                                                             factor=args['factor'],
                                                                             patience=args['patience'],
                                                                             verbose=True,
                                                                             threshold=args['threshold'],
                                                                             threshold_mode='rel',
                                                                             cooldown=0,
                                                                             min_lr=0,
                                                                             eps=1e-08)

    def update_lr(self):
        # Update learning rate, only 'ReduceLROnPlateau' need use the moving train loss
        if self.setting.lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler.step(self.log.moving_train_loss)
        else:
            self.setting.lr_scheduler.step()

    def update_moving_train_loss(self, loss):
        if self.log.moving_train_loss is None:
            self.log.moving_train_loss = loss.item()
        else:
            self.log.moving_train_loss = \
                (1 - self.setting.eps_train_loss) * self.log.moving_train_loss \
                + self.setting.eps_train_loss * loss.item()

    def update_average_statistics(self, loss, phase='train'):
        if phase == 'train':
            self.log.average_train_loss = loss
            if loss < self.log.best_average_train_loss:
                self.log.best_average_train_loss = loss
                self.log.save_status.append('best_train_loss')
            self.log.list_average_train_loss_associate_iter.append([self.log.average_train_loss, self.log.iter])

        elif phase == 'val':
            self.log.average_val_index = loss
            if loss > self.log.best_average_val_index:
                self.log.best_average_val_index = loss
                self.log.save_status.append('best_val_evaluation_index')
            self.log.list_average_val_index_associate_iter.append([self.log.average_val_index, self.log.iter])

    def forward(self, input_, phase):
        time_start_load_data = time.time()
        # To device
        input_ = input_.to(self.setting.device)

        # Record time of moving input from cpu to gpu
        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        # Forward
        if phase == 'train':
            self.setting.optimizer.zero_grad()
        output = self.setting.network(input_)

        return output

    def backward(self, output, target):
        time_start_load_data = time.time()
        for target_i in range(len(target)):
            target[target_i] = target[target_i].to(self.setting.device)

        # Record time of moving target from cpu to gpu
        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        # Optimize
        loss = self.setting.loss_function(output, target)
        loss.backward()
        self.setting.optimizer.step()

        return loss

    def train(self):
        time_start_train = time.time()

        self.setting.network.train()
        sum_train_loss = 0.
        count_iter = 0

        time_start_load_data = time.time()
        for batch_idx, list_loader_output in enumerate(self.setting.train_loader):

            if (self.setting.max_iter is not None) and (self.log.iter >= self.setting.max_iter - 1):
                break
            self.log.iter += 1

            # List_loader_output[0] default as the input
            input_ = list_loader_output['Input'].float().to(self.setting.device)
            target = list_loader_output['GT'].to(self.setting.device)

            # Record time of preparing data
            self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

            # Forward
            output = self.forward(input_, phase='train')

            # Backward
            loss = self.backward(output, target)

            # Used for counting average loss of this epoch
            sum_train_loss += loss.item()
            count_iter += 1

            self.update_moving_train_loss(loss)
            self.update_lr()

            # Print loss during the first epoch
            if self.log.epoch == 0:
                if self.log.iter % 10 == 0:
                    self.print_log_to_file('                Iter %12d       %12.5f\n' %
                                           (self.log.iter, self.log.moving_train_loss), 'a')

            time_start_load_data = time.time()

        if count_iter > 0:
            average_loss = sum_train_loss / count_iter
            self.update_average_statistics(average_loss, phase='train')

        self.time.train_time_per_epoch = time.time() - time_start_train

    def val(self):
        time_start_val = time.time()
        self.setting.network.eval()

        if self.setting.online_evaluation_function_val is None:
            self.print_log_to_file('===============================> No online evaluation method specified ! \n', 'a')
            raise Exception('No online evaluation method specified !')
        else:
            val_index = self.setting.online_evaluation_function_val(self)
            self.update_average_statistics(val_index, phase='val')

        self.time.val_time_per_epoch = time.time() - time_start_val

    def run(self):
        if self.log.iter == -1:
            self.print_log_to_file('Start training !\n', 'w')
        else:
            self.print_log_to_file('Continue training !\n', 'w')
        self.print_log_to_file(time.strftime('Local time: %H:%M:%S\n', time.localtime(time.time())), 'a')

        # Start training
        while (self.log.epoch < self.setting.max_epoch - 1) and (self.log.iter < self.setting.max_iter - 1):
            #
            time_start_this_epoch = time.time()
            self.log.epoch += 1
            # Print current learning rate
            self.print_log_to_file('Epoch: %d, iter: %d\n' % (self.log.epoch, self.log.iter), 'a')
            self.print_log_to_file('    Begin lr is %12.12f, %12.12f\n' % (
                self.setting.optimizer.param_groups[0]['lr'], self.setting.optimizer.param_groups[-1]['lr']), 'a')

            # Record initial learning rate for this epoch
            self.log.list_lr_associate_iter.append([self.setting.optimizer.param_groups[0]['lr'], self.log.iter])

            self.time.__init__()
            self.train()
            self.val()

            # If update learning rate per epoch
            if not self.setting.lr_scheduler_update_on_iter:
                self.update_lr()

            # Save trainer every "self.setting.save_per_epoch"
            if (self.log.epoch + 1) % self.setting.save_per_epoch == 0:
                self.log.save_status.append('iter_' + str(self.log.iter))
            self.log.save_status.append('latest')

            # Try save trainer
            if len(self.log.save_status) > 0:
                for status in self.log.save_status:
                    self.save_trainer(status=status)
                self.log.save_status = []

            self.print_log_to_file(
                '            Average train loss is             %12.12f,     best is           %12.12f\n' %
                (self.log.average_train_loss, self.log.best_average_train_loss), 'a')
            self.print_log_to_file(
                '            Average val evaluation index is   %12.12f,     best is           %12.12f\n'
                % (self.log.average_val_index, self.log.best_average_val_index), 'a')

            self.print_log_to_file('    Train use time %12.5f\n' % (self.time.train_time_per_epoch), 'a')
            self.print_log_to_file('    Train loader use time %12.5f\n' % (self.time.train_loader_time_per_epoch), 'a')
            self.print_log_to_file('    Val use time %12.5f\n' % (self.time.val_time_per_epoch), 'a')
            self.print_log_to_file('    Total use time %12.5f\n' % (time.time() - time_start_this_epoch), 'a')
            self.print_log_to_file('    End lr is %12.12f, %12.12f\n' % (
                self.setting.optimizer.param_groups[0]['lr'], self.setting.optimizer.param_groups[-1]['lr']), 'a')
            self.print_log_to_file(time.strftime('    time: %H:%M:%S\n', time.localtime(time.time())), 'a')

        self.print_log_to_file('===============================> End successfully\n', 'a')

    def print_log_to_file(self, txt, mode):
        with open(self.setting.output_dir + '/log.txt', mode) as log_:
            log_.write(txt)

        # Also display log in the terminal
        txt = txt.replace('\n', '')
        print(txt)

    def save_trainer(self, status='latest'):
        if len(self.setting.list_GPU_ids) > 1:
            network_state_dict = self.setting.network.module.state_dict()
        else:
            network_state_dict = self.setting.network.state_dict()

        optimizer_state_dict = self.setting.optimizer.state_dict()
        lr_scheduler_state_dict = self.setting.lr_scheduler.state_dict()

        ckpt = {
            'network_state_dict': network_state_dict,
            'lr_scheduler_state_dict': lr_scheduler_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'log': self.log
        }

        torch.save(ckpt, self.setting.output_dir + '/' + status + '.pkl')
        self.print_log_to_file('        ==> Saving ' + status + ' model successfully !\n', 'a')

    # Default load trainer in cpu, please reset device using the function self.set_GPU_device
    def init_trainer(self, ckpt_file, list_GPU_ids, only_network=True):
        ckpt = torch.load(ckpt_file, map_location='cpu')

        return ckpt
