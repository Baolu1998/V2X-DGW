import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.V2VAM import V2V_AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder

class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                  kernel_size=1)

        # self.late_fusion_net_cls = V2V_AttFusion(2)
        # self.late_fusion_net_reg = V2V_AttFusion(14)
        
        # self.cls_head_final = nn.Conv2d(args['anchor_number'], args['anchor_number'],
        #                           kernel_size=1)
        # self.reg_head_final = nn.Conv2d(7 * args['anchor_number'], 7 * args['anchor_number'],
        #                           kernel_size=1)
        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.compression_layer = AutoEncoder(384, args['compression'])
        

    def forward(self, data_dict,intermediate_feature=None):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        if 'backup_aug_processed_lidar_torch_dict' in data_dict:
            backup_aug_batch_dict = {'voxel_features': data_dict['backup_aug_processed_lidar_torch_dict']['voxel_features'],
                      'voxel_coords': data_dict['backup_aug_processed_lidar_torch_dict']['voxel_coords'],
                      'voxel_num_points': data_dict['backup_aug_processed_lidar_torch_dict']['voxel_num_points'],
                      'record_len': data_dict['record_len']}
            backup_aug_batch_dict = self.pillar_vfe(backup_aug_batch_dict)
            backup_aug_batch_dict = self.scatter(backup_aug_batch_dict) 

        spatial_features_2d = batch_dict['spatial_features_2d']  #bs,384,100,352

        if intermediate_feature != None:
            spatial_features_2d = intermediate_feature

        if self.compression:
            spatial_features_2d = self.compression_layer(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d) #bs,2,100,352
        #fusion_psm = self.late_fusion_net_cls(psm,record_len)
        rm = self.reg_head(spatial_features_2d) #bs,14,100,352
        #fusion_rm = self.late_fusion_net_reg(rm,record_len)

        # fusion_psm = self.cls_head_final(fusion_psm) #bs,2,100,352
        # fusion_rm = self.reg_head_final(fusion_rm) #bs,14,100,352

        output_dict = {'psm': psm,
                       'rm': rm,
                       'spatial_features_2d':spatial_features_2d,
                       'batch_dict':batch_dict}
        if 'backup_aug_processed_lidar_torch_dict' in data_dict:
             output_dict.update({'back_aug_mask_needs':backup_aug_batch_dict['spatial_features']})

        return output_dict