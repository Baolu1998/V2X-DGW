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
import torch.nn as nn
from opencood.models.domain_generlization.dg_module import Fusion_Level_MetricLearner,Agent_Level_MetricLearner
from opencood.loss.contrastive_loss import SupConLoss,Fusion_CustomContrastiveLoss,Agent_CustomContrastiveLoss
import torch.nn.functional as F



def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', type=str, default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if hypes['DG_params']['Is_DG']:
        if hypes['DG_params']['Fusion_level_metric']: 
            fusion_level_metriclearner_config = hypes['DG_params']['Fusion_level_metric_learner_args']
            fusion_level_metric_learner = Fusion_Level_MetricLearner(fusion_level_metriclearner_config)
            if torch.cuda.is_available():
                fusion_level_metric_learner.to(device)
            fusion_level_metric_func = Fusion_CustomContrastiveLoss()
            fusion_level_metric_weight = hypes['DG_params']['Fusion_level_metric_weight']
        if hypes['DG_params']['Agent_level_metric']: 
            agent_level_metriclearner_config = hypes['DG_params']['Agent_level_metric_learner_args']
            agent_level_metric_learner = Agent_Level_MetricLearner(agent_level_metriclearner_config)
            if torch.cuda.is_available():
                agent_level_metric_learner.to(device)
            agent_level_metric_func = Agent_CustomContrastiveLoss()
            agent_level_metric_weight = hypes['DG_params']['Agent_level_metric_weight']

        if hypes['DG_params']['Agent_level_align']:
            agent_level_align_func = nn.L1Loss()
            agent_level_align_weight = hypes['DG_params']['Agent_level_align_weight']
        if hypes['DG_params']['Fusion_level_align']:
            fusion_level_align_func = nn.L1Loss()
            fusion_level_align_weight = hypes['DG_params']['Fusion_level_align_weight']
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
                                collate_fn=opencood_validate_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)
        print('load model')

    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)

    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module


    criterion = train_utils.create_loss(hypes)


    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)

    if hypes['DG_params']['Is_DG']:
        if hypes['DG_params']['Fusion_level_metric']: 
            optimizer_fusion_metric_learner = train_utils.setup_optimizer(hypes, fusion_level_metric_learner)
        if hypes['DG_params']['Agent_level_metric']: 
            optimizer_agent_metric_learner = train_utils.setup_optimizer(hypes, agent_level_metric_learner)

    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)
    if hypes['DG_params']['Is_DG']:
        if hypes['DG_params']['Fusion_level_metric']: 
            scheduler_fusion_metric_learner = train_utils.setup_lr_schedular(hypes, optimizer_fusion_metric_learner, num_steps)
        if hypes['DG_params']['Agent_level_metric']: 
            scheduler_agent_metric_learner = train_utils.setup_lr_schedular(hypes, optimizer_agent_metric_learner, num_steps)

    writer = SummaryWriter(saved_path)


    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']


    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
            if hypes['DG_params']['Is_DG']:
                if hypes['DG_params']['Fusion_level_metric']: 
                    scheduler_fusion_metric_learner.step(epoch)
                if hypes['DG_params']['Agent_level_metric']: 
                    scheduler_agent_metric_learner.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
            if hypes['DG_params']['Is_DG']:
                if hypes['DG_params']['Fusion_level_metric']: 
                    scheduler_fusion_metric_learner.step_update(epoch * num_steps + 0)
                if hypes['DG_params']['Agent_level_metric']: 
                    scheduler_agent_metric_learner.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)


        #agent_level_align_loss_not_works_time = 0
        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            if hypes['DG_params']['Is_DG']:
                if hypes['DG_params']['Fusion_level_metric']: 
                    fusion_level_metric_learner.train()
                    fusion_level_metric_learner.zero_grad()
                    optimizer_fusion_metric_learner.zero_grad()
                if hypes['DG_params']['Agent_level_metric']:
                    agent_level_metric_learner.train()
                    agent_level_metric_learner.zero_grad()
                    optimizer_agent_metric_learner.zero_grad() 

            cav_masks = batch_data['aug_ego']['cav_masks']
            batch_data['aug_ego'].pop('cav_masks')
            batch_data = train_utils.to_device(batch_data, device)

            if not opt.half:
                source_ouput_dict = model(batch_data['ego'])
                source_final_loss = criterion(source_ouput_dict,
                                       batch_data['ego']['label_dict'],DG_domain='source')

                aug_ouput_dict = model(batch_data['aug_ego'])
                aug_final_loss = criterion(aug_ouput_dict,
                                       batch_data['aug_ego']['label_dict'],DG_domain='aug')
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])

            final_loss = 1*source_final_loss + 1*aug_final_loss
            agent_level_align_loss = None
            if hypes['DG_params']['Agent_level_align']: 
                source_coords = source_ouput_dict['batch_dict']['voxel_coords']
                source_pillars = source_ouput_dict['batch_dict']['pillar_features']
                source_spatial_features = source_ouput_dict['batch_dict']['spatial_features']

                aug_coords = aug_ouput_dict['batch_dict']['voxel_coords']
                aug_pillars = aug_ouput_dict['batch_dict']['pillar_features']
                aug_spatial_features = aug_ouput_dict['batch_dict']['spatial_features']
                backup_aug_spatial_features = aug_ouput_dict['back_aug_mask_needs']

                
                if source_spatial_features.shape[0] == aug_spatial_features.shape[0]:
                    bs, channel, length, width = source_spatial_features.shape
                    mask = torch.zeros(bs, length, width).to(device)
                    non_zero_elements = torch.any(backup_aug_spatial_features != 0, dim=1)
                    mask[non_zero_elements] = 1
                    num = torch.sum(mask)
                    mask = mask.unsqueeze(1).expand(-1, channel, -1, -1)
                    
                    masked_source_spatial = source_spatial_features.detach() * mask
                    agent_level_align_loss = agent_level_align_func(aug_spatial_features,masked_source_spatial)
                    final_loss += agent_level_align_weight * agent_level_align_loss
                else:
                    source_spatial_features = source_spatial_features.detach() 
                    remove_num = 0
                    index = 0
                    for idx in cav_masks:
                        cur_cav_mask = cav_masks[idx]
                        if cur_cav_mask == 0:
                            index_to_remove = index - remove_num
                            source_spatial_features = torch.cat((source_spatial_features[:index_to_remove], source_spatial_features[index_to_remove+1:]), dim=0)
                            remove_num += 1
                        index += 1
                    mask = (aug_spatial_features != 0)
                    if mask.shape == source_spatial_features.shape:
                        
                        bs, channel, length, width = source_spatial_features.shape
                        mask = torch.zeros(bs, length, width).to(device)
                        non_zero_elements = torch.any(backup_aug_spatial_features != 0, dim=1)
                        mask[non_zero_elements] = 1
                        num = torch.sum(mask)
                        mask = mask.unsqueeze(1).expand(-1, channel, -1, -1)

                        masked_source_spatial = source_spatial_features * mask
                        agent_level_align_loss = agent_level_align_func(aug_spatial_features,masked_source_spatial)
                        final_loss += agent_level_align_weight * agent_level_align_loss


            fusion_level_align_loss = None

            if hypes['DG_params']['Fusion_level_align']:  
                source_coords = source_ouput_dict['batch_dict']['voxel_coords']
                source_pillars = source_ouput_dict['batch_dict']['pillar_features']
                source_spatial_features_2d = source_ouput_dict['batch_dict']['spatial_features_2d']

                aug_coords = aug_ouput_dict['batch_dict']['voxel_coords']
                aug_pillars = aug_ouput_dict['batch_dict']['pillar_features']
                aug_spatial_features_2d = aug_ouput_dict['batch_dict']['spatial_features_2d']
                
                
                fusion_level_align_loss = fusion_level_align_func(aug_spatial_features_2d,source_spatial_features_2d.detach())
                final_loss += fusion_level_align_weight * fusion_level_align_loss

            fusion_level_metric_loss = None
            if hypes['DG_params']['Fusion_level_metric']: 
                source_coords = source_ouput_dict['batch_dict']['voxel_coords']
                source_pillars = source_ouput_dict['batch_dict']['pillar_features']
                source_spatial_features_2d = source_ouput_dict['batch_dict']['spatial_features_2d']

                aug_coords = aug_ouput_dict['batch_dict']['voxel_coords']
                aug_pillars = aug_ouput_dict['batch_dict']['pillar_features']
                aug_spatial_features_2d = aug_ouput_dict['batch_dict']['spatial_features_2d']


                source_fusion_features = fusion_level_metric_learner(source_spatial_features_2d)
                aug_fusion_features = fusion_level_metric_learner(aug_spatial_features_2d)
                fusion_level_metric_loss = fusion_level_metric_func(source_fusion_features,aug_fusion_features,manner = 0)
                final_loss += fusion_level_metric_weight * fusion_level_metric_loss


            agent_level_metric_loss = None
            if hypes['DG_params']['Agent_level_metric']: 
                source_coords = source_ouput_dict['batch_dict']['voxel_coords']
                source_pillars = source_ouput_dict['batch_dict']['pillar_features']
                source_spatial_features_2d = source_ouput_dict['batch_dict']['spatial_features_2d']

                aug_coords = aug_ouput_dict['batch_dict']['voxel_coords']
                aug_pillars = aug_ouput_dict['batch_dict']['pillar_features']
                aug_spatial_features_2d = aug_ouput_dict['batch_dict']['spatial_features_2d']

                source_single_features = source_ouput_dict['batch_dict']['before_fusion_features']
                aug_single_features = aug_ouput_dict['batch_dict']['before_fusion_features']

                agent_metric_level = hypes['DG_params']['Agent_metric_level']
                source_single_feature = source_single_features[agent_metric_level]
                aug_single_feature = aug_single_features[agent_metric_level]

                
                if source_single_feature.shape[0] == aug_single_feature.shape[0]:
                    source_agent_features = agent_level_metric_learner(source_single_feature) 

                    aug_agent_features = agent_level_metric_learner(aug_single_feature)
                    cav_ids = list(cav_masks.keys())
                    cav_ids = [int(x) if 'chongfu' not in x else int(x.replace('chongfu', '')) for x in cav_ids]
                    labels = torch.tensor(cav_ids).to(device)
                    agent_level_metric_loss = agent_level_metric_func(source_agent_features,aug_agent_features,labels)
                    final_loss += agent_level_metric_weight * agent_level_metric_loss
                else:
                    source_single_feature = source_single_feature
                    remove_num = 0
                    index = 0
                    for idx in cav_masks:
                        cur_cav_mask = cav_masks[idx]
                        if cur_cav_mask == 0:
                            index_to_remove = index - remove_num
                            source_single_feature = torch.cat((source_single_feature[:index_to_remove], source_single_feature[index_to_remove+1:]), dim=0)
                            remove_num += 1
                        index += 1
                    if source_single_feature.shape[0] == aug_single_feature.shape[0]:
                        source_agent_features = agent_level_metric_learner(source_single_feature)
                        aug_agent_features = agent_level_metric_learner(aug_single_feature)
                        cav_ids = [index for index, value in cav_masks.items() if value != 0]
                        cav_ids = [int(x) if 'chongfu' not in x else int(x.replace('chongfu', '')) for x in cav_ids]
                        labels = torch.tensor(cav_ids).to(device)
                        agent_level_metric_loss = agent_level_metric_func(source_agent_features,aug_agent_features,labels)
                        final_loss += agent_level_metric_weight * agent_level_metric_loss

                        

            criterion.logging_DG(epoch, i, len(train_loader), writer, pbar=pbar2, agent_level_align_loss=agent_level_align_loss,
            fusion_level_align_loss=fusion_level_align_loss,fusion_level_metric_loss=fusion_level_metric_loss,agent_level_metric_loss=agent_level_metric_loss)
            pbar2.update(1)
            
            if not opt.half:
                final_loss.backward()
                optimizer.step()
                if hypes['DG_params']['Is_DG']:
                    if hypes['DG_params']['Fusion_level_metric']: 
                        optimizer_fusion_metric_learner.step()
                    if hypes['DG_params']['Agent_level_metric']: 
                        optimizer_agent_metric_learner.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
                if hypes['DG_params']['Is_DG']:
                    if hypes['DG_params']['Fusion_level_metric']: 
                        scheduler_fusion_metric_learner.step_update(epoch * num_steps + i)
                    if hypes['DG_params']['Agent_level_metric']: 
                        scheduler_agent_metric_learner.step_update(epoch * num_steps + i)
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

            if hypes['DG_params']['Is_DG']:
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
