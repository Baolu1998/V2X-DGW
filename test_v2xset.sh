

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset/point_pillar_intermediate_fusion_2024_02_01_10_06_47'

# CUDA_VISIBLE_DEVICES=2 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method intermediate \
#     --model_dir $model \
#     --isSim 

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset/point_pillar_intermediate_fusion_retrain_14_2024_02_01_10_19_05'

# CUDA_VISIBLE_DEVICES=2 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method intermediate \
#     --model_dir $model \
#     --isSim 

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset/point_pillar_intermediate_fusion_retrain_19_2024_02_01_10_19_50'

# CUDA_VISIBLE_DEVICES=2 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method intermediate \
#     --model_dir $model \
#     --isSim 

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset/point_pillar_intermediate_fusion_retrain_22_2024_02_01_10_20_17'

# CUDA_VISIBLE_DEVICES=2 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method intermediate \
#     --model_dir $model \
#     --isSim 

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset/point_pillar_early_fusion_2024_02_01_10_27_16'

# CUDA_VISIBLE_DEVICES=2 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method early \
#     --model_dir $model \
#     --isSim 

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset/point_pillar_late_fusion_2024_02_01_10_33_59'

# CUDA_VISIBLE_DEVICES=2 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method late \
#     --model_dir $model \
#     --isSim 


# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset_DG/ours_rain'

# CUDA_VISIBLE_DEVICES=4 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference.py \
#     --fusion_method intermediate \
#     --model_dir $model \
#     --isSim 

# model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset_DG/mldg_a'

# CUDA_VISIBLE_DEVICES=4 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference_DG.py \
#     --fusion_method intermediate \
#     --model_dir $model \
#     --isSim 

model='/home/baoluli/personal/2.model_saved/4.Adverseweather/v2xset_DG/mldg_b'

CUDA_VISIBLE_DEVICES=5 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference_DG.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim 