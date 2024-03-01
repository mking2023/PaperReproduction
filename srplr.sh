
seed_lst=(3407)


python_script="main.py"

for seed in "${seed_lst[@]}"; do
	printf "#%.0s" {1..80}
	echo
	echo "Executing $python_script with seed $seed"

	python "$python_script" \
		--lr 0.001 \
		--weight_decay 0.0 \
		--batchsize 512 \
		--batchsize_eval 512 \
		--seed $seed \
		--gpu 3 \
		--max_len 20 \
		--neg_num 100 \
		--dataset "Toys" \
		--model "SRPLR" \
		--base_model "SASRec" \
		--n_layers 2 \
		--n_heads 2 \
		--hidden_size 64 \
		--hidden_dropout_prob 0.5 \
		--attn_dropout_prob 0.2 \
		--tau 0.05 \
		--reg_weight 0.005 \
		--bpr_weight 0.3 \
		--gamma 0.0 \
		--neg_sam_num 3 \
		--full_sort 0 \
		--tune_param 0 \
		--epoch 200 \
		--eval_interval 5 
done
