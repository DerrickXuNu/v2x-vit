import argparse
import os
import statistics
import random

import easydict
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.tools import train_utils
from v2xvit.data_utils.datasets import build_dataset, build_motion_dataset
from v2xvit.tools.train_utils import to_device

DEBUG = True

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")
    parser.add_argument('--stage', type=str, default='stage1', help='Training stage')
    opt = parser.parse_args()
    return opt

def debuging(hypes):
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_train_dataset.__getitem__(0)
    # opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
    print('debuging finished')
    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=True)
    # val_loader = DataLoader(opencood_validate_dataset,
    #                         batch_size=hypes['train_params']['batch_size'],
    #                         num_workers=8,
    #                         collate_fn=opencood_train_dataset.collate_batch_train,
    #                         shuffle=False,
    #                         pin_memory=False,
    #                         drop_last=True)

def main():
    print(os.path.abspath('.'))
    if DEBUG:
        opt = easydict.EasyDict({'hypes_yaml': "/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/point_pillar_early_fusion_vit.yaml",
                                 'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/point_pillar_early_fusion_mswin/',
                                 'half': False,
                                 'stage': 'stage1'})
        # opt = easydict.EasyDict({   'hypes_yaml': "/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/point_pillar_v2xvit_stage3_NotUseRTE_learnable_motion_bs1.yaml",
        #                             'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/with_noise_motion_complete_second_order_simple_learnable_Motion_bs1/',
        #                             'half': False,
        #                             'stage': 'stage3'})
        # opt = easydict.EasyDict({'hypes_yaml': '../../v2xvit/hypes_yaml/point_pillar_v2xvit.yaml',
        #                          'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/motion_second_order_simple',
        #                          'half': False,
        #                          'stage': 'stage2'})
    else:
        opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    stage = opt.stage
    assert stage in ['stage1', 'stage2', 'stage3']
    assert stage == 'stage1' or opt.model_dir, 'stage 2 and 3 must have model_dir'
    print('stage: ', stage)

    set_random_seed(666, True)

    print('Dataset Building')
    # dataset:
    #   scenario_database:
    #       - ordered_dict: every scenario
    #           - ordered_dict: every car (id > 0) or infrastructure (id < 0)
    #               - ordered_dict: every frame (timestamp)
    #                   - yaml: every attribute
    #                   - lidar: (N, 4) numpy array
    #                   - camera:

    # backbone loader
    if stage != 'stage2':
        opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
        opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    if stage == 'stage2':
        # motion loader
        motion_train_dataset = build_motion_dataset(hypes, visualize=False, train=True)
        motion_validate_dataset = build_motion_dataset(hypes, visualize=False, train=False)
        motion_train_loader = DataLoader(motion_train_dataset,
                                         batch_size=hypes['train_params']['batch_size'],
                                         num_workers=8,
                                         collate_fn=motion_train_dataset.collate_batch_train,
                                         shuffle=True,
                                         pin_memory=False,
                                         drop_last=True)
        motion_val_loader = DataLoader(motion_validate_dataset,
                                       batch_size=hypes['train_params']['batch_size'],
                                       num_workers=8,
                                       collate_fn=motion_train_dataset.collate_batch_train,
                                       shuffle=False,
                                       pin_memory=False,
                                       drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    if hypes['use_motion'] or stage == 'stage2':
        motion_model = train_utils.create_model(hypes, stage='stage2')
        print('motion model created')
    else:
        motion_model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        if motion_model is not None:
            motion_model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)
    if stage == 'stage2':
        motion_criterion = train_utils.create_loss(hypes, stage='stage2')

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, motion_model if stage == 'stage2' else model)
    # lr scheduler setup
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    if stage == 'stage3':
        model.load_motion(motion_model)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        # motion_model should be loaded
        init_epoch_motion, motion_model = train_utils.load_saved_model(saved_path,
                                                                       motion_model,
                                                                       # device=device if device == 'cpu' else None,
                                                                       stage='stage2')
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        print('Loaded model from {}, init_epoch = {}'.format(saved_path, init_epoch))
        if init_epoch_motion == 0:
            print('motion model, initEpoch = 0')
        else:
            print('Loaded motion model from {}, init_epoch = {}'.format(saved_path, init_epoch_motion))
        train_utils.save_yaml(hypes, saved_path)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes, opt.hypes_yaml)
        print('Created model folder at {}, initEpoch = 0'.format(saved_path))

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')

    if stage == 'stage2':
        train_stage2(hypes, model, motion_model, motion_train_loader, motion_val_loader, motion_criterion, optimizer,
                     scheduler,
                     writer, device, saved_path, init_epoch_motion)
        return

    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
            # back-propagation
            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        if epoch % hypes['train_params']['eval_freq'] == 0:
            validate_main_model(model, val_loader, criterion, device, writer, epoch)


    # from v2xvit.data_utils.datasets.basedataset import max_timedelay
    # print(max_timedelay)
    # print('max_timedelay', max(max_timedelay))
    print('Training Finished, checkpoints saved to %s' % saved_path)


def train_stage2(hypes, model, motion_model, train_loader, val_loader, criterion, optimizer, scheduler, writer, device,
                 saved_path, init_epoch, scaler=None, half=None):
    print('Training stage 2')
    epochs = hypes['motion_train_params']['epoches']
    for epoch in range(init_epoch, max(init_epoch, epochs)):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):
            motion_model.train()
            model.eval()
            motion_model.zero_grad()
            optimizer.zero_grad()
            batch_data = to_device(batch_data, device)

            # inputs, targets = batch_data
            inputs = batch_data['sources']
            targets = batch_data['target']

            with torch.no_grad():
                history_frames = model.vox2spatial(inputs)
                targets = model.vox2spatial(targets)
            outputs = motion_model(history_frames, batch_data['deltaT'])
            loss = criterion(outputs, targets)
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
            loss.backward()
            optimizer.step()
        if epoch % hypes['motion_train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            with torch.no_grad():
                for j, val_batch_data in enumerate(val_loader):
                    val_batch_data = to_device(val_batch_data, device)
                    val_inputs = val_batch_data['sources']
                    val_targets = val_batch_data['target']
                    val_history_frames = model.vox2spatial(val_inputs)
                    val_targets = model.vox2spatial(val_targets)
                    val_outputs = motion_model(val_history_frames, val_batch_data['deltaT'])
                    val_loss = criterion(val_outputs, val_targets)
                    valid_ave_loss.append(val_loss.item())
            ave_loss = statistics.mean(valid_ave_loss)
            print('Epoch %d, motion validation loss %f' % (epoch, ave_loss))
            writer.add_scalar('motion_val_loss', ave_loss, epoch)
        if epoch % hypes['motion_train_params']['save_freq'] == 0:
            torch.save(motion_model.state_dict(),
                       os.path.join(saved_path, 'motion_epoch%d.pth' % epoch))
    print('Training stage 2 finished, check %s for the saved models' % saved_path)
    return


def validate_main_model(model, val_loader, criterion, device, writer, epoch):
    print('Validating')
    model.eval()
    valid_ave_loss = []
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            model.eval()

            batch_data = train_utils.to_device(batch_data, device)
            ouput_dict = model(batch_data['ego'])

            final_loss = criterion(ouput_dict,
                                   batch_data['ego']['label_dict'])
            valid_ave_loss.append(final_loss.item())
    valid_ave_loss = statistics.mean(valid_ave_loss)
    print('At epoch %d, the validation loss is %f' % (epoch,
                                                      valid_ave_loss))

    writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)


if __name__ == '__main__':
    main()
