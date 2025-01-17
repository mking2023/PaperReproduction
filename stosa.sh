python main.py \
	--epoch 200 \
	--lr 0.001 \
	--weight_decay 0 \
	--batchsize 512 \
	--batchsize_eval 512 \
	--seed 2023 \
	--gpu 0 \
	--max_len 20 \
	--hidden_size 64 \
	--num_hidden_layers 1 \
	--num_attention_heads 4 \
	--attention_probs_dropout_prob 0. \
	--hidden_dropout_prob 0.3 \
	--model 'DistSAModel'  \
	--pvn_weight 0.005 \
	--dataset "Beauty" \
	--full_sort 0 \

