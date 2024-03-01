python main.py \
	--epoch 200 \
	--lr 0.001 \
	--batchsize 512 \
	--batchsize_eval 512 \
	--seed 2023 \
	--gpu 0 \
	--max_len 50 \
	--eval_interval 5 \
	--dataset 'ml-100k' \
	--model 'FMLPRecModel' \
	--emb_dim 128 \
	--num_hidden_layers 4 \
	--num_attention_heads 2 \


