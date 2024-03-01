#!/bin/bash

seed_list=(2023) 
# pattern_level_lst=(6)
# lambda_lst=(0.6)

python_script="main.py"

for seed in "${seed_list[@]}"; do
    
    printf "#%.0s" {1..80}
    echo
    echo "Executing $python_script with seed $seed"
    # echo "Current level => $level"
    # echo "Current lambda => $lambda"

    python "$python_script" \
        --epoch 200 \
        --eval_interval 5 \
        --lr 0.001 \
        --lr_decay_step 1000 \
        --lr_decay_rate 0.75 \
        --batchsize 512 \
        --batchsize_eval 512 \
        --iter_interval 5 \
        --seed $seed \
        --gpu 2 \
        --max_len 20 \
        --emb_dim 64 \
        --base_add 1 \
        --gamma 2 \
        --pattern_level 2 \
        --dataset "Tools" \
        --model "FixedPatternWeightMixer" \
        --emb_type "gamma" \
        --mlp_lambda 0.4 \
        --weight_decay 0.0 \
        --tune_param 0
done
