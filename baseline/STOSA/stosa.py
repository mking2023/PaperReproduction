import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptsr.datasets import *
from ptsr.utils import *
from baseline.STOSA.modules import *


class DistSAModel(nn.Module):
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
        parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
        parser.add_argument('--num_attention_heads', default=2, type=int)
        parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
        parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument('--distance_metric', default='wasserstein', type=str)
        parser.add_argument('--pvn_weight', default=0.1, type=float)
        parser.add_argument('--kernel_param', default=1.0, type=float)        
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
        parser.add_argument('--cuda_condition', type=bool, default=True)
        parser.add_argument('--full_sort', type=int, default=0)
        
        
    def __init__(self, args):
        super(DistSAModel, self).__init__()
        self.item_mean_embeddings = nn.Embedding(args.item_num + 1, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_num + 1, args.hidden_size, padding_idx=0)
        self.position_mean_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.position_cov_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.user_margins = nn.Embedding(args.user_num + 1, 1)
        self.item_encoder = DistSAEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.apply(self.init_weights)


    def add_position_mean_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_mean_embeddings(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        return sequence_emb


    def add_position_cov_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_cov_embeddings(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1

        return sequence_emb


    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)

        mean_sequence_emb = self.add_position_mean_embedding(input_ids)
        cov_sequence_emb = self.add_position_cov_embedding(input_ids)

        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        return mean_sequence_output, cov_sequence_output, att_scores


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    
    def dist_predict_topn(self, seq_mean_out, seq_cov_out, item_indices):
        """
            seq_mean_out : (B, 1, E)
            item_indices : (B, N)
        """
        elu_activation = nn.ELU()    
        test_item_mean_emb = self.item_mean_embeddings(item_indices)  # (B, N, E)
        test_item_cov_emb = elu_activation(self.item_cov_embeddings(item_indices)) + 1
        dis = wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        dis = -dis.squeeze()
        return dis
    
    
    def dist_predict_full(self, seq, seq_mean_out, seq_cov_out):        
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.item_cov_embeddings.weight) + 1
        dis = wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)  # (B, 1, N)
        dis = -dis.squeeze()  # The smaller the distance, the better
        
        update_value = -1e24
        mask = torch.zeros_like(dis)
        mask.scatter_(-1, seq, update_value)
        mask[:, 0] = update_value
        dis = dis + mask
        
        return dis

    
    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):
        # [batch, seq_len, hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.item_cov_embeddings(neg_ids)) + 1

        # [batch * seq_len, hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size)  # [batch * seq_len, hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size)  # [batch * seq_len, hidden_size]
                
        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)            
        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_len).float()  # [batch * seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)
        
        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget) / torch.sum(istarget)

        return loss, auc, pvn_loss
    
    
    def train_model(self, dataset):
        [user_train, user_valid, user_test, user_num, item_num] = dataset
        train_data = ParallelSequentialDataset(user_train, self.args.max_len, self.args.item_num, shuffle=True)        
        valid_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='valid')
        test_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='test')
        
        train_loader = DataLoader(train_data, self.args.batchsize, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_data, self.args.batchsize_eval)
        test_loader = DataLoader(test_data, self.args.batchsize_eval)
        
        best_benchmark = -1.
        final_test_benchmark = None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(self.args.adam_beta1, self.args.adam_beta2))

        for epoch in range(1, self.args.epoch + 1):
            epoch_loss = self.train_an_epoch(optimizer, train_loader)
            print("Epoch = [{}], Loss = {:.5f}, lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr']))
            logging.info("Epoch = [{}], Loss = {:.5f}, lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr']))
            
            if epoch % self.args.eval_interval == 0:
                valid_metric, valid_bench = self.evaluate_model(valid_loader, mode='Valid')
                test_metric, _ = self.evaluate_model(test_loader, mode='Test')

                print(valid_metric)
                print(test_metric)
                logging.info(valid_metric)
                logging.info(test_metric)

                if valid_bench > best_benchmark:
                    best_benchmark = valid_bench
                    final_test_benchmark = test_metric
                    torch.save(self.state_dict(), os.path.join(self.args.log_dir, f'stosa_{self.args.dataset}_epoch_{epoch}.pth'))
        print("=" * 80, '\n', final_test_benchmark)
        logging.info("=" * 100)
        logging.info(final_test_benchmark)

    
    def train_an_epoch(self, optimizer, dataloader):
        self.train()

        epoch_loss = 0.
        for u, seq, pos, neg in add_process_bar(dataloader, desc='Train'):
            optimizer.zero_grad()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            seq, pos, neg = torch.from_numpy(seq), torch.from_numpy(pos), torch.from_numpy(neg)
            seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()

            sequence_mean_output, sequence_cov_output, att_scores = self.finetune(seq)
            loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, pos, neg)
            loss += pvn_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    
    def evaluate_model(self, dataloader, mode='Valid'):
        metric_name = ['NDCG', 'HR']
        topks = [5, 10]
        metric_obj = Metrics(metric_name, topks, is_full=self.args.full_sort, mode=mode, benchmark_name='NDCG', benchmark_k=5)
        
        with torch.no_grad():
            self.eval()
            sample_num = 0
            for seq, pos, neg in add_process_bar(dataloader, desc=mode):
                seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
                item_indices = torch.cat((pos, neg), dim=-1)
                
                seq_mean, seq_cov, _  = self.finetune(seq)
                seq_mean, seq_cov = seq_mean[:, -1, :], seq_cov[:, -1, :]
                seq_mean, seq_cov = seq_mean.unsqueeze(1), seq_cov.unsqueeze(1)
                if self.args.full_sort:
                    scores = self.dist_predict_full(seq, seq_mean, seq_cov)             
                else:
                    scores = self.dist_predict_topn(seq_mean, seq_cov, item_indices)
                metric_obj.metric_value_accumulate(scores, pos)
                sample_num += seq.shape[0]
                break
            
            metric_obj.average_metric_value(sample_num)
            metric_string = metric_obj.metric_value_to_string()
            benchmark_value = metric_obj.get_benchmark_value()
            return metric_string, benchmark_value
                
                