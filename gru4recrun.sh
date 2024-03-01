
seed_lst=(42 2013 2023 2033 3407)


for seed in "${seed_lst[@]}"; do

	printf "#%.0s" {1..80}
	echo
	echo "Current seed => $seed"
	python main.py \
		--epoch 200 \
		--lr 0.001 \
		--batchsize 512 \
		--batchsize_eval 512 \
		--seed $seed \
		--max_len 20 \
		--model 'GRU4Rec' \
		--emb_dim 128 \
		--dropout_rate 0.5 \
		--num_layers 1 \
		--hidden_size 128 \
		--gpu 3 \
		--weight_decay 0.00000001 \
		--dataset "yelp550" \
		--full_sort 0 \
		--tune_param 0 
done
