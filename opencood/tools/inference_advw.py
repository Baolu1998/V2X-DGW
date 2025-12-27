import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from collections import OrderedDict


# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v/point_pillar_late_fusion_2023_12_04_21_59_28'

# CUDA_VISIBLE_DEVICES=0 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method late \
#     --model_dir $model \
#     --isSim 

# def test_parser():
#     parser = argparse.ArgumentParser(description="synthetic data generation")
#     parser.add_argument('--model_dir', type=str,required=True,# default='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v/point_pillar_intermediate_fusion_2023_12_04_22_02_44', #
#                         help='Continued training path')

#     parser.add_argument('--fusion_method',required=True, # default='intermediate', #
#                         type=str,
#                         help='nofusion, late, early or intermediate')
#     parser.add_argument('--show_vis',action='store_true',# default=False,#
#                         help='whether to show image visualization result')
#     parser.add_argument('--show_sequence', action='store_true',
#                         help='whether to show video visualization result.'
#                              'it can note be set true with show_vis together ')
#     parser.add_argument('--save_vis', action='store_true',
#                         help='whether to save visualization result')
#     parser.add_argument('--save_npy', action='store_true',
#                         help='whether to save prediction and gt result'
#                              'in npy file')
#     parser.add_argument('--isSim',action='store_true',# default=True ,#
#                         help='whether to save prediction and gt result'
#                              'in npy file')
#     opt = parser.parse_args()
#     return opt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str,default='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v_advw/unet.yaml', #required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', type=str,default='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v/point_pillar_intermediate_fusion_2023_12_04_22_02_44', #required=True,# 
                        help='Continued training path')
    parser.add_argument('--denoising_model_dir', type=str,default='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v/point_pillar_intermediate_fusion_2023_12_04_22_02_44/unet_2023_12_27_19_16_35', #required=True,# 
                        help='Continued training path')
    parser.add_argument('--fusion_method',default='intermediate', #required=True, # 
                        type=str,
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis',default=False,#action='store_true',# 
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim',default=True ,#action='store_true',# 
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt



def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    #TODO:advw_by_baolu
    denoising_hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    hypes['denoising_params'] = denoising_hypes

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=0,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    denoising_model = train_utils.create_denoising_model(hypes['denoising_params'])
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
        denoising_model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    _, denoising_model = train_utils.load_saved_model(opt.denoising_model_dir, denoising_model)
    
    model.eval()
    denoising_model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())

    for i, batch_data in enumerate(data_loader):
        print(i)
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['processed_lidar'] = batch_data['ego']['advw_processed_lidar']
            if opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            elif opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                output_dict = OrderedDict()
                cav_content = batch_data['ego']

                v1 = model(cav_content)
                intermediate_feature = v1['spatial_features_2d']
                pre = denoising_model(intermediate_feature)
                output_dict['ego'] = model(cav_content,pre)
                pred_box_tensor, pred_score, gt_box_tensor = \
                    opencood_dataset.post_process(batch_data,
                                        output_dict)


                # pred_box_tensor, pred_score, gt_box_tensor = \
                #     infrence_utils.inference_intermediate_fusion(batch_data,
                #                                                  model,
                #                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            # overall calculating
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            # short range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,
                                       left_range=0,
                                       right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.7,
                                       left_range=0,
                                       right_range=30)

            # middle range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,
                                       left_range=30,
                                       right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.7,
                                       left_range=30,
                                       right_range=50)

            # right range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,
                                       left_range=50,
                                       right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.7,
                                       left_range=50,
                                       right_range=100)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
                print('gt:')
                print(gt_box_tensor)
                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.denoising_model_dir)
    eval_utils.eval_final_results(result_stat_short,
                                  opt.denoising_model_dir,
                                  "short")
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.denoising_model_dir,
                                  "middle")
    eval_utils.eval_final_results(result_stat_long,
                                  opt.denoising_model_dir,
                                  "long")
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
