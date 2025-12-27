# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, default='',#required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', type=str, default='',#required=True,#,
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt

# def train_parser():
#     parser = argparse.ArgumentParser(description="synthetic data generation")
#     parser.add_argument("--hypes_yaml", type=str, default='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v_DG/point_pillar_intermediate_fusion_ibnb.yaml',#required=True,
#                         help='data generation yaml file needed ')
#     parser.add_argument('--model_dir', type=str, default='',#required=True,#,
#                         help='Continued training path')
#     parser.add_argument("--half", action='store_true',
#                         help="whether train with half precision.")
#     parser.add_argument('--dist_url', default='env://',
#                         help='url used to set up distributed training')
#     opt = parser.parse_args()
#     return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
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

    #if hypes['DG_params']['Is_DG']:
    validate_num = hypes['DG_params']['DG_validate_num']
    DG_validate_datasets = []
    for item in hypes['DG_params']['DG_validate_dir']:
        DG_validate_datasets.append(build_dataset(hypes, visualize=False, train=False, DG_val_dir=item))
    DG_val_loaders = []
    for item in DG_validate_datasets:
        DG_val_loaders.append(DataLoader(item,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_validate_dataset.collate_batch_train,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=True))


    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)
        print('load model')

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
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
            with open(os.path.join(saved_path,'val_loss.txt'), 'a') as file:
                file.write('At epoch %d, the validation loss is %f \r\n' % (epoch+1,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            #if hypes['DG_params']['Is_DG']:
            all_DG_ave_loss = []
            with torch.no_grad():
                for i in range(hypes['DG_params']['DG_validate_num']):
                    cur_loader = DG_val_loaders[i]
                    cur_valid_losses = []
                    for i, batch_data in enumerate(cur_loader):
                        model.eval()

                        batch_data = train_utils.to_device(batch_data, device)
                        ouput_dict = model(batch_data['ego'])

                        final_loss = criterion(ouput_dict,
                                            batch_data['ego']['label_dict'])
                        cur_valid_losses.append(final_loss.item())
                    cur_valid_ave_loss = statistics.mean(cur_valid_losses)
                    all_DG_ave_loss.append(cur_valid_ave_loss)
            for i in range(hypes['DG_params']['DG_validate_num']):
                cur_valid_set = hypes['DG_params']['DG_validate_name'][i]
                cur_valid_ave_loss = all_DG_ave_loss[i]
                print('At epoch %d, the validation of %s loss is %f' % (epoch,cur_valid_set,
                                                            cur_valid_ave_loss))    

                with open(os.path.join(saved_path,'val_loss.txt'), 'a') as file:
                    file.write('At epoch %d, the validation of %s loss is %f \r\n' % (epoch+1, cur_valid_set,
                                                                cur_valid_ave_loss))


    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
