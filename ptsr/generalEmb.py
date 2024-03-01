import torch
import torch.nn as nn
import torch.nn.functional as F

from PTSR.ptsr.BaseModel import *


class Attention(nn.Module):
    def __init__(self, args) -> None:
        super(Attention, self).__init__()
        self.args = args
        
        self.layer1 = nn.Linear(args.emb_dim, args.emb_dim)
        self.layer2 = nn.Linear(args.emb_dim, args.emb_dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
    
        
    def forward(self, pattern_emb, mask):
        mask = torch.where(mask, 1., -10000.)
        # print("pattern emb = {}".format(pattern_emb.shape))
        layer1_act = F.relu(self.layer1(pattern_emb))   # (..., W, E)
        attention_input = self.layer2(layer1_act) * mask.unsqueeze(-1)  # (..., W, E)
        attention = F.softmax(attention_input, dim=-1)  # (W, E)
        emb = torch.sum(pattern_emb * attention, dim=-2)  # (..., E)
        return emb
            
            
class GeneralEmb(BaseModel):
    
    def __init__(self, args, writer) -> None:
        super(GeneralEmb, self).__init__(args)
        self.args = args
        self.writer = writer

        self.attention = Attention(args)
        self.criterion = nn.BCEWithLogitsLoss()
        self.item_embedding = nn.Embedding(args.item_num + 1, args.emb_dim)

        pattern_num = int(args.pattern_level * (args.max_len - 0.5 * args.pattern_level + 0.5))
        print("pattern num => ",pattern_num)
        self.dense = nn.Linear(in_features=args.emb_dim * args.max_len, out_features=pattern_num)
        
    
    def cal_pattern_target_distance(self, pattern, target):
        """
        Args:
            pattern : (B, X, E)
            target : (B, Y, E)

        Returns:  (B, Y, X)
        """
        return F.cosine_similarity(pattern.unsqueeze(1), target.unsqueeze(2), dim=-1)  # (B, Y, X)
          
    
    def get_pattern_weight(self, seq):
        seq_mask = seq != 0  # (B, S)
        seq_emb = self.item_embedding(seq.long())  # (B, S, E)
        # seq_emb = self.attention(seq_emb, seq_mask)  # (B, E)
        seq_emb = seq_emb.reshape(seq_emb.shape[0], -1)  # (B, S * E)
        pattern_w = F.softmax(self.dense(seq_emb), dim=-1)  # (B, N)
        return pattern_w
        
    
    def forward(self, seq, pos, neg):
        pos_emb = self.item_embedding(pos.long())  # (B, 1, E)
        neg_emb = self.item_embedding(neg.long())  # (B, Y, E)
        
        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_mask_lst = []
        for i in range(1, self.args.pattern_level + 1):
            pattern, mask = self.get_pattern_index(seq, window_size=i)  # (B, X, W)
            pattern_mask_lst.append(torch.max(mask, dim=-1)[0])  # (B, X)
            pattern_emb = self.item_embedding(pattern.long())  # (B, X, W, E)
            pattern_emb = self.attention(pattern_emb, mask)  # (B, X, E)
            pattern_pos_dis = self.cal_pattern_target_distance(pattern_emb, pos_emb)  # (B, 1, X)
            pattern_neg_dis = self.cal_pattern_target_distance(pattern_emb, neg_emb)  # (B, Y, X)
            pos_pattern_dis_lst.append(pattern_pos_dis)
            neg_pattern_dis_lst.append(pattern_neg_dis)
        pos_dis = torch.cat(pos_pattern_dis_lst, dim=-1)  # (B, 1, N)
        neg_dis = torch.cat(neg_pattern_dis_lst, dim=-1)  # (B, Y, N)
        pattern_mask = torch.cat(pattern_mask_lst, dim=-1)  # (B, N)
        pattern_weight = self.get_pattern_weight(seq)  # (B, N)
        pattern_weight = (pattern_weight * pattern_mask).unsqueeze(1)  # (B, 1, N)
        pos_score = torch.sum(pos_dis * pattern_weight, dim=-1)  # (B, 1)
        neg_score = torch.sum(neg_dis * pattern_weight, dim=-1)  # (B, Y)
        
        # pos_score = torch.sum(pos_dis * pattern_mask.unsqueeze(1), dim=-1)  # (B, 1)
        # neg_score = torch.sum(neg_dis * pattern_mask.unsqueeze(1), dim=-1)  # (B, Y)
        
        return pos_score, neg_score, None
    
    
    def predict(self, seq, pos, neg):
        pos_score, neg_score, _ = self.forward(seq, pos, neg)  # (B, 1)  (B, Y)
        scores = torch.cat((pos_score, neg_score), dim=-1)  # (B, 1 + Y)
        return scores, _
    
    
    def calculate_loss(self, seq, pos, neg):
        pos_score, neg_score, _ = self.forward(seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_score.shape).cuda(), torch.zeros(neg_score.shape).cuda()
        loss = self.criterion(pos_score, pos_labels)
        loss += self.criterion(neg_score, neg_labels)
        return loss