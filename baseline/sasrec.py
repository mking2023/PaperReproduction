import numpy as np
import torch
import copy
import wandb
from ptsr.utils import *
from tqdm import tqdm
from ptsr.datasets import *


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument('--hidden_units', type=int, default=50)
        parser.add_argument('--num_blocks', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=1)
        parser.add_argument('--dropout_rate', type=float, default=0.5)
        parser.add_argument('--l2_emb', type=float, default=0.)
                        
    
    def __init__(self, args, writer):
        super(SASRec, self).__init__()

        self.user_num = None
        self.item_num = args.item_num
        self.args = args
        self.writer = writer

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_len, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()


    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).cuda())
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        
        cnt = 0
        first_layer_attn_weight = None
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_weight = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            
            if cnt == 0:
                cnt += 1
                first_layer_attn_weight = attn_weight[:, -1]
                        
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats, first_layer_attn_weight


    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats, _ = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits  # pos_pred, neg_pred


    def predict(self, log_seqs, item_indices):  # for inference
        log_feats, first_weight = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste  (U, C)
        item_embs = self.item_emb(item_indices)  # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits, first_weight  # preds # (U, I)
    
    
    def predict_full(self, log_seqs):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # (B, E)
        item_embs = self.item_emb.weight  # (N, E)
        logits = torch.matmul(final_feat, item_embs.transpose(-1, -2))  # (B, N)

        update_value = -1e24
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, log_seqs, update_value)
        mask[:, 0] = update_value
        logits = logits + mask
        
        return logits
                   
    
    def train_model(self, dataset):
        [user_train, user_valid, user_test, user_num, item_num] = dataset
        train_data = ParallelSequentialDataset(user_train, self.args.max_len, self.args.item_num, shuffle=True)        
        valid_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='valid')
        test_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='test')
        
        train_loader = DataLoader(train_data, self.args.batchsize, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_data, self.args.batchsize_eval)
        test_loader = DataLoader(test_data, self.args.batchsize_eval)
        
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass
        
        best_valid_bench = -1
        final_test_bench = -1.
        final_test_bench_string = None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), weight_decay=self.args.weight_decay)
        bce_criterion = nn.BCEWithLogitsLoss()

        for epoch in range(1, self.args.epoch + 1):
            epoch_loss = self.train_an_epoch(optimizer, bce_criterion, train_loader)
            if self.writer != None:
                self.writer.add_scalar('Loss', epoch_loss, epoch)
            print("Epoch = [{}], Loss = {:.5f}, lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr']))
            logging.info("Epoch = [{}], Loss = {:.5f}, lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr']))
            if epoch % self.args.eval_interval == 0:
                valid_metric, valid_bench = self.evaluate_model(epoch, valid_loader, mode='Valid')
                test_metric, test_bech = self.evaluate_model(epoch, test_loader, mode='Test')
                
                print(valid_metric)
                print(test_metric)
                logging.info(valid_metric)
                logging.info(test_metric)                  
                
                if valid_bench > best_valid_bench:
                    best_valid_bench = valid_bench
                    final_test_bench = test_bech
                    final_test_bench_string = test_metric
                    torch.save(self.state_dict(), os.path.join(self.args.log_dir, f'sasrec_{self.args.dataset}_epoch_{epoch}.pth'))
        
        
        print("=" * 80, '\n', final_test_bench_string)
        logging.info("=" * 100)
        logging.info(final_test_bench_string)
        return best_valid_bench, final_test_bench
    
    
    def train_an_epoch(self, optimizer, criterion, dataloader):
        self.train()

        epoch_loss = 0.
        for u, seq, pos, neg in add_process_bar(dataloader, desc='Train'):
            optimizer.zero_grad()
            
            u, seq, pos, neg = u.cuda(), seq.cuda(), pos.cuda(), neg.cuda()
            pos_logits, neg_logits = self.forward(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()
            indices = np.where(pos.detach().cpu().numpy() != 0)
            
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])
            for param in self.item_emb.parameters():
                loss += self.args.l2_emb * torch.norm(param)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(dataloader)
        return epoch_loss
          
        
    def evaluate_model(self, epoch, dataloader, mode='Valid'):
        metric_name = ['NDCG', 'HR']
        topks = [5, 10]
        metric_obj = Metrics(metric_name, topks, is_full=self.args.full_sort, mode=mode, benchmark_name='NDCG', benchmark_k=10)
        
        with torch.no_grad():
            self.eval()
            sample_num = 0
            for uid, seq, pos, neg in add_process_bar(dataloader, desc=mode):
                seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
                item_indices = torch.cat((pos, neg), dim=-1)
                if self.args.full_sort:
                    scores = self.predict_full(seq)
                else:
                    scores, _ = self.predict(seq, item_indices)
                sample_num += seq.shape[0]
                metric_obj.metric_value_accumulate(scores, pos)                
            
            metric_obj.average_metric_value(sample_num)
            metric_string = metric_obj.metric_value_to_string()
            benchmark_value = metric_obj.get_benchmark_value()
            if self.writer != None:
                self.writer.add_scalar(f'Metric/{mode}_NDCG10', metric_obj.metric_dict['NDCG'][10], epoch)
            
            return metric_string, benchmark_value
        
    
    def visualize_weight(self, dataset, save_path=None):
        user_train, user_valid, user_test, user_num, item_num = dataset
        test_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='test')
        test_loader = DataLoader(test_data, self.args.batchsize_eval)
        
        metric_name = ['NDCG', 'HR']
        topks = [5, 10]
        metric_obj = Metrics(metric_name, topks, is_full=self.args.full_sort, mode='Test', benchmark_name='NDCG', benchmark_k=10)
        
        all_uid_lst = []
        all_seq_pos_lst = []
        all_weight_lst = []
        
        with torch.no_grad():
            self.eval()
            sample_num = 0
            for uid, seq, pos, neg in add_process_bar(test_loader, desc='Test'):
                uid, seq, pos, neg = uid.cuda(), seq.cuda(), pos.cuda(), neg.cuda()
                item_indices = torch.cat((pos, neg), dim=-1)
                scores, attn_weight = self.predict(seq, item_indices)  # (B, I)  (B, S)
                sample_num += seq.shape[0]
                metric_obj.metric_value_accumulate(scores, pos)                
                
                all_uid_lst.append(uid.unsqueeze(-1))
                all_seq_pos_lst.append(torch.cat((seq, pos), dim=-1))
                all_weight_lst.append(attn_weight)
                
            # metric_obj.average_metric_value(sample_num)
            # metric_string = metric_obj.metric_value_to_string()
            # benchmark_value = metric_obj.get_benchmark_value()
            # print(metric_string)
            
            all_user = torch.cat(all_uid_lst, dim=0)
            all_seq_pos = torch.cat(all_seq_pos_lst, dim=0)
            all_weight = torch.cat(all_weight_lst, dim=0)
            
            print(all_user.shape)
            print(all_seq_pos.shape)
            print(all_weight.shape)
            
            all_user_seq = torch.cat((all_user, all_seq_pos), dim=-1).cpu().numpy()
            all_user_weight = torch.cat((all_user, all_weight), dim=-1).cpu().numpy()
            print("all user seq = ", all_user_seq.shape)
            print("all user weigth = ", all_user_weight.shape)
            
            start_time = time.time()
            user_seq_path = os.path.join(save_path, './user_seq.txt')
            user_weight_path = os.path.join(save_path, './user_weight.txt')
            np.savetxt(user_seq_path, all_user_seq, fmt='%d', delimiter='\t')
            np.savetxt(user_weight_path, all_user_weight, fmt='%.4f', delimiter='\t')
            end_time = time.time()
            print("Save Time Consume: [{:.2f}]".format(end_time - start_time))
            
            
    def visualize_weight(self, dataset, save_path=None):
        user_train, user_valid, user_test, user_num, item_num = dataset
        test_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='test')
        test_loader = DataLoader(test_data, self.args.batchsize_eval)
        
        metric_name = ['NDCG', 'HR']
        topks = [5, 10]
        metric_obj = Metrics(metric_name, topks, is_full=False, mode='Test', benchmark_name='NDCG', benchmark_k=10)
        
        all_uid_lst = []
        all_seq_pos_lst = []
        all_pos_weight_lst = [] 
        all_score_lst = []
        all_sample_lst = []
        
        with torch.no_grad():
            self.eval()
            sample_num = 0
            cnt = 0
            for user, seq, pos, neg in add_process_bar(test_loader, desc='Test'):
                user, seq, pos, neg = user.cuda(), seq.cuda(), pos.cuda(), neg.cuda()
                item_indices = torch.cat((pos, neg), dim=-1)  # (B, Y)
                scores, attn_weight = self.predict(seq, item_indices)
                sample_num += seq.shape[0]
                metric_obj.metric_value_accumulate(scores)
            
                all_uid_lst.append(user.unsqueeze(-1))
                all_seq_pos_lst.append(torch.cat((seq, pos), dim=-1))
                all_pos_weight_lst.append(attn_weight.squeeze(1))
                all_score_lst.append(scores)  # (B, Y)
                all_sample_lst.append(torch.cat((pos, neg), dim=-1))  # (B, Y)
                # cnt += 1
                # if cnt > 2:
                #     break
                
            all_user = torch.cat(all_uid_lst, dim=0)
            all_seq_pos = torch.cat(all_seq_pos_lst, dim=0)
            all_pos_weight = torch.cat(all_pos_weight_lst, dim=0)
            all_score = torch.cat(all_score_lst, dim=0)
            all_sample = torch.cat(all_sample_lst, dim=0)
            
            print(all_user.shape)
            print(all_seq_pos.shape)
            print(all_pos_weight.shape)
            print(all_score.shape)
            print(all_sample.shape)
            
            all_user_seq = torch.cat((all_user, all_seq_pos), dim=-1).cpu().numpy()
            all_user_weight = torch.cat((all_user, all_pos_weight), dim=-1).cpu().numpy()
            all_user_score = torch.cat((all_user, all_score), dim=-1).cpu().numpy()
            all_user_sample = torch.cat((all_user, all_sample), dim=-1).cpu().numpy()
            
            print("all user seq = ", all_user_seq.shape)
            print("all user weigth = ", all_user_weight.shape)
            print("all user score = ", all_user_score.shape)
            print("all user sample = ", all_user_sample.shape)
            
            start_time = time.time()
            user_seq_path = os.path.join(save_path, './user_seq.txt')
            user_weight_path = os.path.join(save_path, './user_weight.txt')
            user_score_path = os.path.join(save_path, './user_score.txt')
            user_sample_path = os.path.join(save_path, './user_sample.txt')
            
            np.savetxt(user_seq_path, all_user_seq, fmt='%d', delimiter='\t')
            np.savetxt(user_weight_path, all_user_weight, fmt='%.4f', delimiter='\t')
            np.savetxt(user_score_path, all_user_score, fmt='%.4f', delimiter='\t')
            np.savetxt(user_sample_path, all_user_sample, fmt='%.4f', delimiter='\t')
            end_time = time.time()
            print("Save Time Consume: [{:.2f}]".format(end_time - start_time))
            
            # metric_obj.average_metric_value(sample_num)
            # metric_string = metric_obj.metric_value_to_string()
            # benchmark_value = metric_obj.get_benchmark_value()
            # print(metric_string)