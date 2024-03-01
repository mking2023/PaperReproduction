python main_case_study.py \
	--epoch 200 \
	--eval_interval 5 \
	--lr 0.001 \
	--lr_decay_step 1000 \
	--lr_decay_rate 0.75 \
	--batchsize 512 \
	--batchsize_eval 512 \
	--iter_interval 5 \
	--seed 2023 \
	--gpu 2 \
	--max_len 10 \
	--emb_dim 64 \
	--base_add 1 \
	--gamma 2 \
	--pattern_level 2 \
	--dataset "Beauty" \
	--model "FixedPatternWeightMixer" \
	--emb_type "gamma" \
	--mlp_lambda 0.4 \
	--weight_decay 0.0000001 \
	--tune_param 0
