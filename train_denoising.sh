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

hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/opv2v_advw/unet.yaml'
model='/home/baoluli/personal/2.model_saved/4.Adverseweather/opv2v/point_pillar_intermediate_fusion_2023_12_04_22_02_44'


CUDA_VISIBLE_DEVICES=2 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/train_advw.py --hypes_yaml $hypes_yaml --model_dir $model 
