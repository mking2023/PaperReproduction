import torch
import torch.nn as nn
import torch.nn.functional as F

from PTSR.ptsr.BaseModel import *


class PatternProjection(BaseModel):
    def __init__(self, args, writer) -> None:
        super(PatternProjection, self).__init__(args)
        self.args = args
        self.writer = writer
        self.weight_tau = args.weight_tau
        self.mlp_lambda = args.mlp_lambda
        self.exp_lambda = args.exp_lambda

        for i in range(1, self.args.pattern_level + 1):
            setattr(self, f'projection_level{i}', GammaProjection(args))
        
        for i in range(1, self.args.pattern_level + 1):
            cur_pattern_num = self.args.max_len - i + 1
            cur_pattern_dim = self.args.emb_dim
            setattr(self, f'weight_level{i}', nn.Linear(in_features=cur_pattern_dim * cur_pattern_num, out_features=cur_pattern_num))
        
        self.get_pattern_num()
        
    
    def get_pattern_weight_per_level(self, pattern_alpha, pattern_beta, pattern_mask, level):
        """
            pattern_alpha : (B, X, E)
            pattern_beta : (B, Y, E)
            pattern_mask : (B, X)
        """
        pattern_emb = pattern_alpha / (pattern_alpha + pattern_beta)  # (B, X, E)
        pattern_emb = pattern_emb * pattern_mask.unsqueeze(-1)
        pattern_emb = pattern_emb.reshape(pattern_emb.shape[0], -1)  # (B, X * E)
        
        layer = getattr(self, f'weight_level{level}')
        pattern_w = layer(pattern_emb)  # (B, X)
        
        weight_mask = torch.where(pattern_mask, 0., -10000)
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
        # weight = torch.sigmoid(pattern_target_dis)  # (B, Y, X)
        # weight = torch.tanh(pattern_target_dis) + 1# (B, Y, X)
        return weight
    
    
    def distance_to_exp_weight(self, pattern_target_dis, mask):
        """
            pattern_target_dis: (B, Y, X)
            mask: (B, X)
        """
        pattern_target_dis = pattern_target_dis - self.args.gamma  # (B, Y, X)
        dis_mask = torch.where(mask, 0., -10000.)  # (B, X)
        pattern_target_dis = pattern_target_dis + dis_mask.unsqueeze(1)  # (B, Y, X)
        max_value = torch.max(pattern_target_dis, dim=-1)[0].unsqueeze(-1)  # (B, Y, 1)
        exp_term = (max_value - pattern_target_dis) / pattern_target_dis  # (B, Y, X)
        exp_weight = torch.exp(exp_term * self.exp_lambda)  # (B, Y, X)
        exp_weight = exp_weight * mask.unsqueeze(1)  # (B, Y, X)
        return exp_weight

    
    def get_weight_bias_per_level(self, level, mask):
        """
            mask : (B, X)
        """
        weight = getattr(self, f'weight_bias_level{level}')  # (1, X)
        weight_mask = torch.where(mask > 0, 0., -10000.)  # (B, X)
        weight = weight + weight_mask  # (B, X)
        weight = F.softmax(weight, dim=-1)  # (B, X)
        return weight
        
    
    def get_pattern_num(self):
        seq_len = self.args.max_len
        lev_num = self.args.pattern_level
        pattern_num = int(lev_num * (2 * seq_len - lev_num + 1) * 0.5)
        print("Pattern type = [sliding], pattern_num = [{}]".format(pattern_num))
        logging.info("Pattern type = [sliding], pattern_num = [{}]".format(pattern_num))
    
    
    def accumulate_pattern_dis_weight(self, seq, pos, neg):
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
            # pattern_alpha, pattern_beta = self.gammaProjection(pattern_alpha, pattern_beta)  # (B, X, E)
            pattern_alpha, pattern_beta = getattr(self, f'projection_level{i}')(pattern_alpha, pattern_beta)
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
    
    
    def accumulate_pattern_mlp_weight(self, seq, pos, neg):
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
    
    
    def accumulate_pattern_dis_mlp_weight(self, seq, pos, neg):
        pos_alpha, pos_beta = self.get_embedding(pos)  # (B, 1, E)
        neg_alpha, neg_beta = self.get_embedding(neg)  # (B, Y, E)
        
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
            cur_both_weight = self.get_pattern_weight_per_level(pattern_alpha, pattern_beta, pattern_mask_lst[-1], level=i)  # (B, X)
            
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

        # logging.info("pos dis shape = {}".format(pos_dis.shape))
        # logging.info(pos_dis[0])            
        # logging.info("both weight shape = {}".format(both_weight.shape))
        # logging.info(both_weight[0])
        # logging.info("pos weight shape = {}".format(pos_weight.shape))
        # logging.info(pos_weight[0])
        
        pos_weight = pos_weight + both_weight.unsqueeze(1) * self.mlp_lambda
        neg_weight = neg_weight + both_weight.unsqueeze(1) * self.mlp_lambda
        
        # logging.info("After Adding")
        # logging.info(pos_weight[0])
        # logging.info("=" * 80)
        
        pos_score = torch.sum(pos_weight * pos_dis, dim=-1)  # (B, 1) 
        neg_score = torch.sum(neg_weight * neg_dis, dim=-1)  # (B, Y) 
        return pos_score, neg_score, pos_weight
    
    
    def accumulate_pattern_dis_exp_weight(self, seq, pos, neg):
        pos_alpha, pos_beta = self.get_embedding(pos)  # (B, 1, E)
        neg_alpha, neg_beta = self.get_embedding(neg)  # (B, Y, E)
        
        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_mask_lst = []
        pos_weight_lst = []
        neg_weight_lst = []
        
        for i in range(1, self.args.pattern_level + 1):            
            pattern, mask = self.get_pattern_index(seq, window_size=i)  # (B, X, W)  (B, X, W)
                   
            pattern_mask_lst.append(torch.max(mask, dim=-1)[0])  # (B, X)
            pattern_alpha, pattern_beta = self.get_embedding(pattern)  # (B, X, W, E)
            
            pattern_alpha, pattern_beta = self.intersection(pattern_alpha, pattern_beta, mask)  # (B, X, E)
            pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha, pos_beta)  # (B, 1, X)
            pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha, neg_beta)  # (B, Y, X)
            cur_pos_weight = self.distance_to_exp_weight(pattern_pos_dis, pattern_mask_lst[-1])  # (B, 1, X)
            cur_neg_weight = self.distance_to_exp_weight(pattern_neg_dis, pattern_mask_lst[-1])  # (B, Y, X)

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
        logging.info(pos_weight[0][0])
        logging.info("neg weight shape = {}".format(neg_weight.shape))
        logging.info(neg_weight[0][0])
        logging.info("=" * 80)
        
        pos_score = torch.sum(pos_weight * pos_dis, dim=-1)  # (B, 1) 
        neg_score = torch.sum(neg_weight * neg_dis, dim=-1)  # (B, Y) 
        return pos_score, neg_score


    def forward(self, seq, pos, neg):         
        pos_score, neg_score = self.accumulate_pattern_dis_weight(seq, pos, neg)
        # pos_score, neg_score = self.accumulate_pattern_mlp_weight(seq, pos, neg)
        # pos_score, neg_score, pos_weight = self.accumulate_pattern_dis_mlp_weight(seq, pos, neg)
        # pos_score, neg_score = self.accumulate_pattern_dis_bias_weight(seq, pos, neg)
        # pos_score, neg_score = self.accumulate_pattern_dis_exp_weight(seq, pos, neg)
        return pos_score, neg_score, None
    
    
    def predict(self, seq, pos, neg):
        pos_score, neg_score, _ = self.forward(seq, pos, neg)  # (B, 1)  (B, Y)  (B, N)
        scores = torch.cat((pos_score, neg_score), dim=-1)  # (B, 1 + Y)
        return scores, _
    
    
    def calculate_loss(self, seq, pos, neg):
        pos_score, neg_score, _ = self.forward(seq, pos, neg)
        loss = torch.sum(-torch.log(torch.sigmoid(pos_score)) - torch.log(torch.sigmoid(-neg_score))) / seq.shape[0]
        return loss
