import os 
import logging
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from ptsr.datasets import *
from ptsr.utils import *
from recbole.model.loss import BPRLoss


class GRU4Rec(nn.Module):
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument('--emb_dim', type=int, default=64)
        parser.add_argument('--dropout_rate', type=float, default=0.3)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--hidden_size', type=int, default=128)

    
    def __init__(self, args, writer) -> None:
        super(GRU4Rec, self).__init__()
        self.args = args
        self.writer = writer
        
        self.item_embedding = nn.Embedding(self.args.item_num + 1, self.args.emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.args.dropout_rate)
        self.gru_layers = nn.GRU(
            input_size=self.args.emb_dim,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            bias=False,
            batch_first=True,
        )        
        self.dense = nn.Linear(self.args.hidden_size, self.args.emb_dim)

        self.loss_fct = BPRLoss()
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            nn.init.xavier_uniform_(module.weight_hh_l0)
            nn.init.xavier_uniform_(module.weight_ih_l0)
    
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    
    def forward(self, seq, item_seq_len):
        seq_emb = self.item_embedding(seq)
        seq_emb_dropout = self.emb_dropout(seq_emb)
        gru_output, _ = self.gru_layers(seq_emb_dropout)
        gru_output = self.dense(gru_output)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)  # (B, E)
        return seq_output
    

    def predict(self, seq, item_indices, item_seq_len):
        log_feats = self.forward(seq, item_seq_len)  # (B, E)
        item_embs = self.item_embedding(item_indices)  # (B, I, E)
        logits = item_embs.matmul(log_feats.unsqueeze(-1)).squeeze(-1)  # (B, I)
        return logits
    
    
    def predict_full(self, seq, item_seq_len):
        seq_output = self.forward(seq, item_seq_len)  # (B, E)
        item_emb = self.item_embedding.weight  # (B, I, E)
        logits = torch.matmul(seq_output, item_emb.transpose(-1, -2))  # (B, I)
        
        update_value = -1e24
        mask = torch.zeros_like(logits) 
        mask.scatter_(-1, seq, update_value)
        mask[:, 0] = update_value
        logits = logits + mask
        return logits
        
    
    def calculate_loss(self, seq, pos, neg, item_seq_len):
        seq_output = self.forward(seq, item_seq_len)  # (B, E)
        pos_emb = self.item_embedding(pos)  # (B, 1, E)
        neg_emb = self.item_embedding(neg)  # (B, 1, E)
        pos_score = torch.sum(seq_output * pos_emb.squeeze(1), dim=-1)  # (B)
        neg_score = torch.sum(seq_output * neg_emb.squeeze(1), dim=-1)  # (B)
        loss = self.loss_fct(pos_score, neg_score)
        return loss
    
    
    def train_model(self, dataset):
        user_train, user_valid, user_test, user_num, item_num = dataset
        train_data = SequentialDataset(self.args, user_train, padding_mode='right')
        valid_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='valid', padding_mode='right')
        test_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='test', padding_mode='right')
        
        trainloader = DataLoader(train_data, self.args.batchsize, shuffle=True)
        validloader = DataLoader(valid_data, self.args.batchsize_eval)
        testloader = DataLoader(test_data, self.args.batchsize_eval)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        best_valid_bench = -1. 
        final_test_bench = -1. 
        final_test_bench_string = -1.
        
        for epoch in range(1, self.args.epoch + 1):
            epoch_loss = self.train_an_epoch(trainloader, optimizer)
            out_str = "Epoch [{}], Loss = {:.5f}, lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr'])
            print(out_str)
            logging.info(out_str)
            if self.writer != None:
                self.writer.add_scalar('Loss', epoch_loss, epoch)
            
            if epoch % self.args.eval_interval == 0:
                valid_metric, valid_bench = self.evaluate_model(epoch, validloader, mode='Valid')
                test_metric, test_bench = self.evaluate_model(epoch, testloader, mode='Test')
                
                print(valid_metric)
                print(test_metric)
                logging.info(valid_metric)
                logging.info(test_metric)              
                
                if valid_bench > best_valid_bench:
                    best_valid_bench = valid_bench
                    final_test_bench = test_bench
                    final_test_bench_string = test_metric
                    torch.save(self.state_dict(), os.path.join(self.args.log_dir, f'{self.args.model}_{self.args.dataset}_epoch_{epoch}.pth'))
        
        print("=" * 80, '\n', final_test_bench_string)
        logging.info("=" * 100)
        logging.info(final_test_bench_string)
        return best_valid_bench, final_test_bench
        
    
    def train_an_epoch(self, trainloader, optimizer):
        self.train()
        epoch_loss = []
        
        for seq, pos, neg, item_seq_len in add_process_bar(trainloader, desc='Train'):
            optimizer.zero_grad()
            seq, pos, neg, item_seq_len = seq.cuda(), pos.cuda(), neg.cuda(), item_seq_len.cuda()
            loss = self.calculate_loss(seq, pos, neg, item_seq_len)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        return np.mean(epoch_loss)
    
    
    def evaluate_model(self, epoch, dataloader, mode='Valid'):
        metric_name = ['NDCG', 'HR']
        topks = [5, 10]
        metric_obj = Metrics(metric_name, topks, is_full=self.args.full_sort, mode=mode, benchmark_name='NDCG', benchmark_k=10)
                
        with torch.no_grad():
            self.eval()
            sample_num = 0
            for user, seq, pos, neg, item_seq_len in add_process_bar(dataloader, desc=mode):
                seq, pos, neg, item_seq_len = seq.cuda(), pos.cuda(), neg.cuda(), item_seq_len.cuda()
                item_indices = torch.cat((pos, neg), dim=-1)  # (B, I)
                if self.args.full_sort:
                    scores = self.predict_full(seq, item_seq_len)
                else:
                    scores = self.predict(seq, item_indices, item_seq_len)
                sample_num += seq.shape[0]
                metric_obj.metric_value_accumulate(scores)
            
            metric_obj.average_metric_value(sample_num)
            metric_string = metric_obj.metric_value_to_string()
            benchmark_value = metric_obj.get_benchmark_value()
            
            if self.writer != None:
                for name in metric_name:
                    for k in topks:
                        self.writer.add_scalar(f'Metric/{mode}_{name}{k}', metric_obj.metric_dict[name][k], epoch)
                
            return metric_string, benchmark_value
            
                
                
            