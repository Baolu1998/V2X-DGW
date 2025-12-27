# hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2xset_DG/point_pillar_intermediate_fusion_ibnb.yaml'
# CUDA_VISIBLE_DEVICES=7 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/train_DG_ibn.py --hypes_yaml $hypes_yaml #--model_dir $model


# hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2xset_DG/point_pillar_intermediate_fusion_mldg(a).yaml'
# CUDA_VISIBLE_DEVICES=6 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/train_DG_mldg_a.py --hypes_yaml $hypes_yaml #--model_dir $model


hypes_yaml='/home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/hypes_yaml/v2xset_DG/point_pillar_intermediate_fusion_mldg(b).yaml'
CUDA_VISIBLE_DEVICES=5 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/train_DG_mldg_b.py --hypes_yaml $hypes_yaml #--model_dir $model