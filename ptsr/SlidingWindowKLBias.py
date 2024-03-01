import torch
import torch.nn as nn
import torch.nn.functional as F

from KMRec.ptsr.BaseModel import *


class PatternWeightNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PatternWeightNet, self).__init__()
        
        self.mlp = nn.Linear(input_dim, output_dim)    
    
    
    def forward(self, seq, seq_alpha, seq_beta, pattern_mask):
        """
            seq: (B, S)
            seq_alpha: (B, S, E)
            seq_beta: (B, S, E)
            pattern_mask: (B, X)
        """
        mask = seq != 0  # (B, S)
        seq_emb = seq_alpha / (seq_alpha + seq_beta)  # (B, S, E)
        seq_emb = seq_emb * mask.unsqueeze(-1)  # (B, S, E)
               
        inputs = seq_emb.reshape(seq_emb.shape[0], -1)  # (B, S * E)
        
        outputs = self.mlp(inputs)  # (B, X)
        weight_mask = torch.where(pattern_mask, 0., -10000.)  # (B, X)
        outputs = outputs + weight_mask
        outputs = F.softmax(outputs, dim=-1)  # (B, X)
        return outputs        


class SlidingWindowKlBias(BaseModel):
    def __init__(self, args, writer) -> None:
        super(SlidingWindowKlBias, self).__init__(args)
        self.args = args
        self.writer = writer
        self.weight_tau = args.weight_tau
        self.mlp_lambda = args.mlp_lambda

        self.get_pattern_num()
        
        for i in range(1, self.args.pattern_level + 1):
            cur_pattern_num = self.args.max_len - i + 1
            cur_input_dim = self.args.emb_dim * self.args.max_len
            cur_weight_net = PatternWeightNet(args, cur_input_dim, cur_pattern_num, num_layers=2, hidden_dim=512)
            setattr(self, f'weight_net_level{i}', cur_weight_net)

        self.apply(self._init_weight)
    
    
    def get_sequence_emb(self, seq):
        seq_mask = seq != 0  # (B, S)
        seq_alpha, seq_beta = self.get_embedding(seq)  # (B, S, E)
        seq_alpha, seq_beta = seq_alpha * seq_mask.unsqueeze(-1), seq_beta * seq_mask.unsqueeze(-1) 
        seq_alpha = self.sequenceMixer(seq_alpha)  # (B, S, E)
        seq_beta = self.sequenceMixer(seq_beta)  # (B, S, E)
        seq_alpha = seq_alpha[:, -1]  # (B, E)
        seq_beta = seq_beta[:, -1]  # (B, E)
        return seq_alpha, seq_beta
    
    
    def get_pattern_weight_per_level(self, seq_alpha, seq_beta, pattern_alpha, pattern_beta, mask, level):
        """
            seq_alpha : (B, E)
            seq_beta : (B, E)
            pattern_alpha : (B, X, E)
            pattern_beta : (B, Y, E)
            mask : (B, X)
        """
        pattern_alpha = torch.sum(pattern_alpha * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=-1).unsqueeze(-1)  # (B, E)
        pattern_beta = torch.sum(pattern_beta * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=-1).unsqueeze(-1)  # (B, E)
        inputs = torch.cat((seq_alpha, seq_beta, pattern_alpha, pattern_beta), dim=-1)  # (B, E * 4)
        
        layer = getattr(self, f'weight_level{level}')
        pattern_w = layer(inputs)  # (B, X)
        
        weight_mask = torch.where(mask, 0., -10000)
        pattern_w = pattern_w + weight_mask
        pattern_w = F.softmax(pattern_w, dim=-1)  # (B, X)
        return pattern_w    
        
    
    def distance_to_weight(self, pattern_target_dis, mask):
        """
            pattern_target_dis: (B, Y, X)
            mask: (B, X)
        """
        attention_mask = torch.where(mask, 0, -10000).unsqueeze(1)  # (B, 1, X)
        pattern_target_dis = pattern_target_dis + attention_mask  # (B, Y, X)
        weight = F.softmax(pattern_target_dis / self.weight_tau, dim=-1)  # (B, Y, X)
        return weight  
        
    
    def get_pattern_num(self):
        seq_len = self.args.max_len
        lev_num = self.args.pattern_level
        pattern_num = int(lev_num * (2 * seq_len - lev_num + 1) * 0.5)
        print("Pattern type = [sliding], pattern_num = [{}]".format(pattern_num))
        logging.info("Pattern type = [sliding], pattern_num = [{}]".format(pattern_num))
    
    
    def accumulate_pattern_kl_weight(self, seq, pos, neg):
        pos_alpha, pos_beta = self.get_embedding(pos)  # (B, 1, E)
        neg_alpha, neg_beta = self.get_embedding(neg)  # (B, Y, E)
        
        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_mask_lst = []
        pos_weight_lst = []
        neg_weight_lst = []
        
        for i in range(1, self.args.pattern_level + 1):            
            pattern, mask = self.get_pattern_index(seq, window_size=i )  # (B, X, W)  (B, X, W)
                   
            pattern_mask_lst.append(torch.max(mask, dim=-1)[0])  # (B, X)
            pattern_alpha, pattern_beta = self.get_embedding(pattern)  # (B, X, W, E)
            
            pattern_alpha, pattern_beta = self.intersection(pattern_alpha, pattern_beta, mask)  # (B, X, E)
            pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha, pos_beta)  # (B, 1, X)
            pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha, neg_beta)  # (B, Y, X)
            cur_pos_weight = self.distance_to_weight(pattern_pos_dis, pattern_mask_lst[-1])  # (B, 1, X)
            cur_neg_weight = self.distance_to_weight(pattern_neg_dis, pattern_mask_lst[-1])  # (B, Y, X)

            pos_weight_lst.append(cur_pos_weight)
            neg_weight_lst.append(cur_neg_weight)
            pos_pattern_dis_lst.append(pattern_pos_dis)
            neg_pattern_dis_lst.append(pattern_neg_dis)
            
        pos_dis = torch.cat(pos_pattern_dis_lst, dim=-1)  # (B, 1, N)
        neg_dis = torch.cat(neg_pattern_dis_lst, dim=-1)  # (B, Y, N)
        pos_weight = torch.cat(pos_weight_lst, dim=-1)  # (B, 1, N)
        neg_weight = torch.cat(neg_weight_lst, dim=-1)  # (B, Y, N)
        
        logging.info("pos dis shape = {}".format(pos_dis.shape))
        logging.info(pos_dis[0])
        logging.info("pos weight shape = {}".format(pos_weight.shape))
        logging.info(pos_weight[0])
        logging.info("=" * 80)
        
        pos_score = torch.sum(pos_weight * pos_dis, dim=-1)  # (B, 1) 
        neg_score = torch.sum(neg_weight * neg_dis, dim=-1)  # (B, Y) 
        return pos_score, neg_score
    
    
    def accumulate_pattern_bias_weight(self, seq, pos, neg):
        pos_alpha, pos_beta = self.get_embedding(pos)  # (B, 1, E)
        neg_alpha, neg_beta = self.get_embedding(neg)  # (B, Y, E)
        
        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_weight_lst = []
        pattern_mask_lst = []
        
        for i in range(1, self.args.pattern_level + 1):            
            pattern, mask = self.get_pattern_index(seq, window_size=i )  # (B, X, W)  (B, X, W)
                   
            pattern_mask_lst.append(torch.max(mask, dim=-1)[0])  # (B, X)
            pattern_alpha, pattern_beta = self.get_embedding(pattern)  # (B, X, W, E)
            
            pattern_alpha, pattern_beta = self.intersection(pattern_alpha, pattern_beta, mask)  # (B, X, E)
            pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha, pos_beta)  # (B, 1, X)
            pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha, neg_beta)  # (B, Y, X)
            cur_pattern_weight = self.get_pattern_weight_per_level(pattern_alpha, pattern_beta, pattern_mask_lst[-1], level=i)  # (B, X)

            pos_pattern_dis_lst.append(pattern_pos_dis)
            neg_pattern_dis_lst.append(pattern_neg_dis)
            pattern_weight_lst.append(cur_pattern_weight)    
            
        pos_dis = torch.cat(pos_pattern_dis_lst, dim=-1)  # (B, 1, N)
        neg_dis = torch.cat(neg_pattern_dis_lst, dim=-1)  # (B, Y, N)
        pattern_weight = torch.cat(pattern_weight_lst, dim=-1)  # (B, N)
        pattern_weight = pattern_weight.unsqueeze(1)  # (B, 1, N)
        pos_score = torch.sum(pos_dis * pattern_weight, dim=-1)  # (B, 1)
        neg_score = torch.sum(neg_dis * pattern_weight, dim=-1)  # (B, Y)
        
        logging.info("pos dis shape = {}".format(pos_dis.shape))
        logging.info(pos_dis[0])
        logging.info("pattern weight shape = {}".format(pattern_weight.shape))
        logging.info(pattern_weight[0])
        logging.info("=" * 80)
        
        return pos_score, neg_score
    
    
    def accumulate_pattern_kl_bias_weight(self, seq, pos, neg):
        pos_alpha, pos_beta = self.get_embedding(pos)  # (B, 1, E)
        neg_alpha, neg_beta = self.get_embedding(neg)  # (B, Y, E)
        # seq_alpha, seq_beta = self.get_sequence_emb(seq)  # (B, E)
        seq_alpha, seq_beta = self.get_embedding(seq)  # (B, S, E)
        
        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_mask_lst = []
        pos_weight_lst = []
        neg_weight_lst = []
        both_weight_lst = []
        
        for i in range(1, self.args.pattern_level + 1):            
            pattern, mask = self.get_pattern_index(seq, window_size=i )  # (B, X, W)  (B, X, W)
                   
            pattern_mask_lst.append(torch.max(mask, dim=-1)[0])  # (B, X)
            pattern_alpha, pattern_beta = self.get_embedding(pattern)  # (B, X, W, E)
            
            pattern_alpha, pattern_beta = self.intersection(pattern_alpha, pattern_beta, mask)  # (B, X, E)
            pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha, pos_beta)  # (B, 1, X)
            pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha, neg_beta)  # (B, Y, X)
            cur_pos_weight = self.distance_to_weight(pattern_pos_dis, pattern_mask_lst[-1])  # (B, 1, X)
            cur_neg_weight = self.distance_to_weight(pattern_neg_dis, pattern_mask_lst[-1])  # (B, Y, X)
            # cur_both_weight = getattr(self, f'weight_net_level{i}')(seq_alpha, seq_beta, pattern_alpha, pattern_beta, pattern_mask_lst[-1])
            cur_both_weight = getattr(self, f'weight_net_level{i}')(seq, seq_alpha, seq_beta, pattern_mask_lst[-1])  # (B, X)
            
            pos_weight_lst.append(cur_pos_weight)
            neg_weight_lst.append(cur_neg_weight)
            both_weight_lst.append(cur_both_weight)
            pos_pattern_dis_lst.append(pattern_pos_dis)
            neg_pattern_dis_lst.append(pattern_neg_dis)
            
        pos_dis = torch.cat(pos_pattern_dis_lst, dim=-1)  # (B, 1, N)
        neg_dis = torch.cat(neg_pattern_dis_lst, dim=-1)  # (B, Y, N)
        pos_weight = torch.cat(pos_weight_lst, dim=-1)  # (B, 1, N)
        neg_weight = torch.cat(neg_weight_lst, dim=-1)  # (B, Y, N)
        both_weight = torch.cat(both_weight_lst, dim=-1)  # (B, N)
        
        pos_weight = pos_weight + both_weight.unsqueeze(1) * self.mlp_lambda
        neg_weight = neg_weight + both_weight.unsqueeze(1) * self.mlp_lambda  # (B, Y, N)     
        pos_score = torch.sum(pos_weight * pos_dis, dim=-1)  # (B, 1) 
        neg_score = torch.sum(neg_weight * neg_dis, dim=-1)  # (B, Y)
            
        return pos_score, neg_score, pos_weight 
      
    
    def accumulate_pattern_without_weight(self, seq, pos, neg):
        pos_alpha, pos_beta = self.get_embedding(pos)  # (B, 1, E)
        neg_alpha, neg_beta = self.get_embedding(neg)  # (B, Y, E)

        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_mask_lst = []
        
        for i in range(1, self.args.pattern_level + 1):            
            pattern, mask = self.get_pattern_index(seq, window_size=i )  # (B, X, W)  (B, X, W)
                   
            pattern_mask_lst.append(torch.max(mask, dim=-1)[0])  # (B, X)
            pattern_alpha, pattern_beta = self.get_embedding(pattern)  # (B, X, W, E)
            
            pattern_alpha, pattern_beta = self.intersection(pattern_alpha, pattern_beta, mask)  # (B, X, E)
            pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha, pos_beta)  # (B, 1, X)
            pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha, neg_beta)  # (B, Y, X)

            pos_pattern_dis_lst.append(pattern_pos_dis)
            neg_pattern_dis_lst.append(pattern_neg_dis)
            
        pos_dis = torch.cat(pos_pattern_dis_lst, dim=-1)  # (B, 1, N)
        neg_dis = torch.cat(neg_pattern_dis_lst, dim=-1)  # (B, Y, N)
  
        pos_score = torch.sum(pos_dis, dim=-1)  # (B, 1) 
        neg_score = torch.sum(neg_dis, dim=-1)  # (B, Y) 
        return pos_score, neg_score, None
    

    def forward(self, seq, pos, neg):         
        pos_score, neg_score, pos_weight = self.accumulate_pattern_kl_bias_weight(seq, pos, neg)
        return pos_score, neg_score, pos_weight
    
    
    def predict(self, seq, pos, neg):
        pos_score, neg_score, pos_weight = self.forward(seq, pos, neg)  # (B, 1)  (B, Y)  (B, N)
        scores = torch.cat((pos_score, neg_score), dim=-1)  # (B, 1 + Y)
        return scores, pos_weight
    
    
    def calculate_loss(self, seq, pos, neg):
        pos_score, neg_score, _ = self.forward(seq, pos, neg)
        loss = torch.sum(-torch.log(torch.sigmoid(pos_score)) - torch.log(torch.sigmoid(-neg_score))) / seq.shape[0]
        return loss


    def case_study(self, seq, pos, neg):
        pos_score, neg_score, pos_weight, pos_dis, pos_weight_dis, neg_dis = self.accumulate_pattern_kl_bias_weight(seq, pos, neg)  # (B, 1)  (B, Y)  (B, 1, N)
        scores = torch.cat((pos_score, neg_score), dim=-1)  # (B, 1 + Y)
        return scores, pos_weight, pos_dis, pos_weight_dis, neg_dis