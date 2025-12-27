#opv2v
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_early_fusion.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_late_fusion_v2.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_intermediate_fusion.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_cobevt.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_fcooper.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_v2vnet_v2.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_v2xvit.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_where2comm.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_intermediate_V2VAM.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v/point_pillar_S2R_UViT.yaml'

#v2v4real
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_early_fusion.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_late_fusion.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_intermediate_fusion.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_cobevt.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_fcooper.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_v2vnet.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_v2xvit.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_where2comm.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_intermediate_V2VAM.yaml'
#hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real/point_pillar_S2R_UViT.yaml'

#model='/home/baoluli/personal/2.model_saved/4.Adverseweather/V2V4Real/point_pillar_S2R_UViT_2023_12_11_12_40_45'

hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2v4real_DG/base.yaml'
CUDA_VISIBLE_DEVICES=7 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/train.py --hypes_yaml $hypes_yaml #--model_dir $model 
