
model='/home/baoluli/personal/2.model_saved/4.Adverseweather/ablation_study/mldga/1'

CUDA_VISIBLE_DEVICES=6 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference_DG_ablationstudy.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim 

model='/home/baoluli/personal/2.model_saved/4.Adverseweather/ablation_study/mldga/2'

CUDA_VISIBLE_DEVICES=6 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference_DG_ablationstudy.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim 

model='/home/baoluli/personal/2.model_saved/4.Adverseweather/ablation_study/mldga/3'

CUDA_VISIBLE_DEVICES=6 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference_DG_ablationstudy.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim 

model='/home/baoluli/personal/2.model_saved/4.Adverseweather/ablation_study/mldga/4'

CUDA_VISIBLE_DEVICES=6 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference_DG_ablationstudy.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim 

model='/home/baoluli/personal/2.model_saved/4.Adverseweather/ablation_study/mldga/5'

CUDA_VISIBLE_DEVICES=6 python3 /home/baoluli/1.code/4.Adverseweather/V2V_baolu/opencood/tools/inference_DG_ablationstudy.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim 
