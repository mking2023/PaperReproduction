#!/bin/bash

seed_list=(2023)

python_script="main.py"

for seed in "${seed_list[@]}"; do
    
    printf "#%.0s" {1..80}
    echo
    echo "Executing $python_script with seed $seed"

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
        --gpu 1 \
        --max_len 20 \
        --emb_dim 128 \
        --base_add 1 \
        --gamma 2 \
        --pattern_level 2 \
        --dataset "yelp550" \
        --model "GeneralEmb" \
        --emb_type "general" \
        --weight_decay 0.0 \
        --tune_param 0
done
