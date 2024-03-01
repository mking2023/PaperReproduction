# -*- coding: utf-8 -*-
# @Time   : 2020/10/2
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
NextItNet
################################################

Reference:
    Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation" in WSDM 2019.

Reference code:
    - https://github.com/fajieyuan/nextitnet
    - https://github.com/initlisk/nextitnet_pytorch

"""
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_
from recbole.model.loss import RegLoss, BPRLoss

from baseline.NextItNet.modules import *
from ptsr.datasets import *
from ptsr.utils import *


class NextItNet(nn.Module):
    r"""The network architecture of the NextItNet model is formed of a stack of holed convolutional layers, which can
    efficiently increase the receptive fields without relying on the pooling operation.
    Also residual block structure is used to ease the optimization for much deeper networks.

    Note:
        As paper said, for comparison purpose, we only predict the next one item in our evaluation,
        and then stop the generating process. Although the number of parameters in residual block (a) is less
        than it in residual block (b), the performance of b is better than a.
        So in our model, we use residual block (b).
        In addition, when dilations is not equal to 1, the training may be slow. To  speed up the efficiency, please set the parameters "reproducibility" False.
    """
    @staticmethod
    def parse_args(parser):
        parser.add_argument('--embedding_size', type=int, default=64)
        parser.add_argument('--block_num', type=int, default=5)
        parser.add_argument('--dilations', type=str, default='1@4')
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--reg_weight', type=float, default=0.)
        parser.add_argument('--full_sort', type=int, default=0)
        

    def __init__(self, args, writer):
        super(NextItNet, self).__init__()
        
        self.args = args
        self.embedding_size = args.embedding_size
        self.residual_channels = args.embedding_size
        self.block_num = args.block_num
        self.dilations = list(map(int, args.dilations.split('@'))) * self.block_num
        self.kernel_size = args.kernel_size
        self.reg_weight = args.reg_weight
        self.item_num = args.item_num
        self.writer = writer        


        # define layers and loss
        self.item_embedding = nn.Embedding(self.item_num + 1, self.embedding_size, padding_idx=0)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels,
                self.residual_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
            )
            for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.embedding_size)

        # Loss function
        self.loss_fct = BPRLoss()
        self.reg_loss = RegLoss()

        # parameters initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1.0 / self.item_num)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)


    def forward(self, item_seq):
        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        seq_output = self.final_layer(hidden)  # [batch_size, embedding_size]
        return seq_output


    def reg_loss_rb(self):
        r"""
            L2 loss on residual blocks
        """
        loss_rb = 0
        if self.reg_weight > 0.0:
            for name, parm in self.residual_blocks.named_parameters():
                if name.endswith("weight"):
                    loss_rb += torch.norm(parm, 2)
        return self.reg_weight * loss_rb
    
    
    def calculate_loss(self, seq, pos, neg):
        """
            seq: (B, S)
            pos: (B, 1)
            neg: (B, 1)
        """
        seq_output = self.forward(seq)  # (B, E)
        pos_emb = self.item_embedding(pos)  # (B, 1, E)
        neg_emb = self.item_embedding(neg)  # (B, 1, E)
        pos_score = torch.sum(seq_output * pos_emb.squeeze(1), dim=-1)  # (B)
        neg_score = torch.sum(seq_output * neg_emb.squeeze(1), dim=-1)  # (B)
        loss = self.loss_fct(pos_score, neg_score)
        reg_loss = self.reg_loss([self.item_embedding.weight, self.final_layer.weight])
        loss = loss + self.reg_weight * reg_loss + self.reg_loss_rb()
        return loss
    

    def predict(self, seq, pos, neg):
        seq_output = self.forward(seq)  # (B, E)
        test_item = torch.cat((pos, neg), dim=-1)  # (B, I)
        test_item_emb = self.item_embedding(test_item)  # (B, I, E)
        scores = test_item_emb.matmul(seq_output.unsqueeze(-1)).squeeze(-1)  # (B, I)
        return scores


    def predict_full(self, seq):
        seq_output = self.forward(seq)  # (B, E)
        item_emb = self.item_embedding.weight  # (I, E)
        logits = torch.matmul(seq_output, item_emb.transpose(-1, -2))  # (B, I)
        
        update_value = -1e24
        mask = torch.zeros_like(logits) 
        mask.scatter_(-1, seq, update_value)
        mask[:, 0] = update_value
        logits = logits + mask
        return logits
    
    
    def train_model(self, dataset):
        user_train, user_valid, user_test, user_num, item_num = dataset
        train_data = SequentialDataset(self.args, user_train, padding_mode='Left')
        valid_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='valid', padding_mode='left')
        test_data = EvalDataset2(self.args, user_train, user_valid, user_test, mode='test', padding_mode='left')
        
        train_loader = DataLoader(train_data, self.args.batchsize, shuffle=True)
        valid_loader = DataLoader(valid_data, self.args.batchsize_eval)
        test_loader = DataLoader(test_data, self.args.batchsize_eval) 
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
                
        best_benchmark = -1
        final_test_benchmark = None
        
        for epoch in range(1, self.args.epoch + 1):
            epoch_loss = self.train_an_epoch(train_loader, optimizer)    
            out_str = "Epoch [{}] Loss = {:.5f} lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr'])
            print(out_str)
            logging.info(out_str)
            self.writer.add_scalar('Loss', epoch_loss, epoch)
            
            if epoch % self.args.eval_interval == 0:
                valid_metric, valid_bench = self.evaluate_model(epoch, valid_loader, mode='Valid')
                test_metric, _ = self.evaluate_model(epoch, test_loader, mode='Test')
                
                print(valid_metric)
                print(test_metric)
                logging.info(valid_metric)
                logging.info(test_metric)              
                
                if valid_bench > best_benchmark:
                    best_benchmark = valid_bench
                    final_test_benchmark = test_metric
                    torch.save(self.state_dict(), os.path.join(self.args.log_dir, f'{self.args.model}_{self.args.dataset}_epoch_{epoch}.pth'))
        
        print("=" * 80, '\n', final_test_benchmark)
        logging.info("=" * 100)
        logging.info(final_test_benchmark)

    
    def train_an_epoch(self, train_loader, optimizer):
        epoch_loss = []
        self.train()
        for seq, pos, neg in add_process_bar(train_loader, desc='Train'):
            optimizer.zero_grad()
            seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
            loss = self.calculate_loss(seq, pos, neg)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        return np.mean(epoch_loss)
    
    
    def evaluate_model(self, epoch, dataloader, mode='Valid'):
        metric_name = ['NDCG', 'HR']
        topks = [5, 10]
        metric_obj = Metrics(metric_name, topks, is_full=False, mode=mode, benchmark_name='NDCG', benchmark_k=10)
                
        with torch.no_grad():
            self.eval()
            sample_num = 0
            for user, seq, pos, neg in add_process_bar(dataloader, desc=mode):
                seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
                if self.args.full_sort:
                    scores = self.full_sort_predict(seq)
                else:
                    scores = self.predict(seq, pos, neg)   # (B, I)
                sample_num += seq.shape[0]
                metric_obj.metric_value_accumulate(scores)
            
            metric_obj.average_metric_value(sample_num)
            metric_string = metric_obj.metric_value_to_string()
            benchmark_value = metric_obj.get_benchmark_value()
            
            for name in metric_name:
                for k in topks:
                    self.writer.add_scalar(f'Metric/{mode}_{name}{k}', metric_obj.metric_dict[name][k], epoch)
            
            return metric_string, benchmark_value

