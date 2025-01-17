python main.py \
	--embedding_size 128 \
	--block_num 5 \
	--dilations "1@4" \
	--kernel_size 3 \
	--reg_weight 0. \
	--epoch 200 \
	--lr 0.001 \
	--batchsize 512 \
	--batchsize_eval 512 \
	--seed 2023 \
	--gpu 1 \
	--max_len 20 \
	--neg_num 100 \
	--dataset "Beauty" \
	--model "NextItNet" \
	--tune_param 0 \
	--full_sort 1 \
