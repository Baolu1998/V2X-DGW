import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import time

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
import statistics

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v/point_pillar_late_fusion_2023_12_04_21_59_28'

# CUDA_VISIBLE_DEVICES=0 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method late \
#     --model_dir $model \
#     --isSim 

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,required=True,# default='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v/point_pillar_intermediate_fusion_2023_12_04_22_02_44', #
                        help='Continued training path')
    parser.add_argument('--fusion_method',required=True, # default='intermediate', #
                        type=str,
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis',action='store_true',# default=False,#
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim',action='store_true',# default=True ,#
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


# def test_parser():
#     parser = argparse.ArgumentParser(description="synthetic data generation")
#     parser.add_argument('--model_dir', type=str,default='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v_DG/point_pillar_intermediate_fusion_all_2024_02_22_13_13_43', #required=True,# 
#                         help='Continued training path')
#     parser.add_argument('--fusion_method',default='intermediate', #required=True, # 
#                         type=str,
#                         help='nofusion, late, early or intermediate')
#     parser.add_argument('--show_vis',default=False,#action='store_true',# 
#                         help='whether to show image visualization result')
#     parser.add_argument('--show_sequence', default=False,#action='store_true',
#                         help='whether to show video visualization result.'
#                              'it can note be set true with show_vis together ')
#     parser.add_argument('--save_vis', action='store_true',
#                         help='whether to save visualization result')
#     parser.add_argument('--save_npy', action='store_true',
#                         help='whether to save prediction and gt result'
#                              'in npy file')
#     parser.add_argument('--isSim',default=True ,#action='store_true',# 
#                         help='whether to save prediction and gt result'
#                              'in npy file')
#     opt = parser.parse_args()
#     return opt



def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    criterion = train_utils.create_loss(hypes)
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=False, train=False,
                                     isSim=opt.isSim)
    val_loader = DataLoader(opencood_dataset,
                             batch_size=2,
                             num_workers=8,
                             collate_fn=opencood_dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)



    #if hypes['DG_params']['Is_DG']:
    validate_num = hypes['DG_params']['DG_validate_num']
    DG_validate_datasets = []
    for item in hypes['DG_params']['DG_validate_dir']:
        DG_validate_datasets.append(build_dataset(hypes, visualize=False, train=False, DG_val_dir=item))
    DG_val_loaders = []
    for item in DG_validate_datasets:
        DG_val_loaders.append(DataLoader(item,
                            batch_size=2,
                            num_workers=8,
                            collate_fn=item.collate_batch_train,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False))

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    
    import glob
    file_list = glob.glob(os.path.join(saved_path, '*epoch*.pth'))
    sorted_file_paths = sorted(file_list, key=lambda x: int(x.split('_epoch')[-1].split('.')[0]))

    for model_index in range(10,len(sorted_file_paths)):
        epoch = model_index + 1
        cur_model_path = sorted_file_paths[model_index]
        model.load_state_dict(torch.load(
            os.path.join(cur_model_path)), strict=False)


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

        with open(os.path.join(saved_path,'full_range_val_loss.txt'), 'a') as file:
            file.write('At epoch %d, the validation loss is %f \r\n' % (epoch,
                                                            valid_ave_loss))
 
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

            with open(os.path.join(saved_path,'full_range_val_loss.txt'), 'a') as file:
                file.write('At epoch %d, the validation of %s loss is %f \r\n' % (epoch+1, cur_valid_set,
                                                            cur_valid_ave_loss))




if __name__ == '__main__':
    main()
