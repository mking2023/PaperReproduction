
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import RegLoss, BPRLoss
from recbole.sampler.sampler import AbstractSampler

from ptsr.datasets import *
from ptsr.utils import *


class Intersection(nn.Module):
    def __init__(self, dim, tau):
        super(Intersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.feature_layer_2 = nn.Linear(2 * self.dim, self.dim, bias=False)
        self.tau = tau
        # self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim, bias=False)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, alpha, beta, mask):
        # feature: N x B x d
        # logic:  N x B x d
        logits = torch.cat([alpha, beta], dim=-1)  # N x B x 2d
        # mask is needed
        mask = torch.where(mask, 0.0, -10000.0)
        # print(mask[0, 0:5])
        # att_input = self.feature_layer_2(F.relu(self.feature_layer_1(logits))) * mask.unsqueeze(2) * 0.05
        att_input = self.feature_layer_1(logits) * mask.unsqueeze(2) * self.tau
        # att_input = self.feature_layer_1(logits) * mask.unsqueeze(2) * 0.05
        # print('att_input', att_input[0, 0:5, 0])

        attention = F.softmax(att_input, dim=1)
        # print('att', attention[0, 0:5, 0])

        alpha = torch.sum(attention * alpha, dim=1)
        beta = torch.sum(attention * beta, dim=1)

        # alpha, beta = self.

        return alpha, beta


class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_feature(self, feature):
        # f,f' in [-L, L]
        # f' = (f + 2L) % (2L) - L, where L=1
        feature = feature
        # indicator_positive = feature >= 0
        # indicator_negative = feature < 0
        # feature[indicator_positive] = feature[indicator_positive] - 1
        # feature[indicator_negative] = feature[indicator_negative] + 1
        return feature

    # def forward(self, feature, logic):
    #     feature = self.neg_feature(feature)
    #     logic = 1 - logic
    #     return feature, logic

    def forward(self, logic):
        logic = 1./logic
        return logic

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class LogicalSampler(AbstractSampler):
    def __init__(self, args, distribution, alpha=1.0):
        self.user_num = args.user_num
        self.item_num = args.item_num
        super().__init__(distribution=distribution, alpha=alpha)


    def _uni_sampling(self, seq_num):
        return np.random.randint(1, self.item_num, seq_num)


    def get_used_ids(self):
        pass


    def sample_neg_sequence(self, pos_sequence, sample_num):
        """For each moment, sampling 'sample_num' item from all the items except the one the user clicked on at that moment.

        Args:
            pos_sequence (torch.Tensor):  all users' item history sequence, with the shape of `(N, )`.

        Returns:
            torch.tensor : all users' negative item history sequence.

        """
        total_num = len(pos_sequence)
        value_ids = []
        for i in range(sample_num):
            check_list = np.arange(total_num)
            tem_ids = np.zeros(total_num, dtype=np.int64)
            while len(check_list) > 0:
                tem_ids[check_list] = self._uni_sampling(len(check_list))
                check_index = np.where(tem_ids[check_list] == pos_sequence[check_list])
                check_list = check_list[check_index]
            pos_sequence = torch.cat([torch.tensor(tem_ids.reshape(total_num, 1)).cuda(), pos_sequence], dim=-1)
            value_ids.append(tem_ids)

        value_ids = torch.tensor(np.array(value_ids)).t().cuda()
        return value_ids



class SRPLR(nn.Module):
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument('--base_model', type=str, default='SASRec')
        parser.add_argument('--n_layers', type=int, default=2)
        parser.add_argument('--n_heads', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)
        parser.add_argument('--attn_dropout_prob', type=float, default=0.5)
        parser.add_argument('--inner_size', type=int, default=256)
        parser.add_argument('--hidden_act', type=str, default='gelu')
        parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
        parser.add_argument('--initializer_range', type=float, default=0.02)
        parser.add_argument('--tau', type=float, default=0.05)
        parser.add_argument('--reg_weight', type=float, default=5e-3)
        parser.add_argument('--bpr_weight', type=float, default=0.5)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--neg_sam_num', type=int, default=3)
        

    def __init__(self, args, writer):
        super(SRPLR, self).__init__()
        
        self.base_model = args.base_model 
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.hidden_size = args.hidden_size 
        self.inner_size = args.inner_size
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.attn_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.reg_weight = args.reg_weight
        self.bpr_weight = args.bpr_weight
        self.tau = args.tau    
        self.neg_sam_num = args.neg_sam_num 
        self.item_num = args.item_num
        self.max_len = args.max_len     
        self.loss_type = 'BPR'   
        self.rec_loss_type = 'CE'
        self.writer = writer
        self.args = args
        
        
        # sasrec base
        self.item_embedding = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.logic_CE = nn.CrossEntropyLoss()
        self.logic_BCE = nn.BCEWithLogitsLoss()
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()
        self.reg_loss = RegLoss()

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # logic initialization
        self.Negation = Negation()
        self.intersection = Intersection(self.hidden_size, self.tau)
        self.gamma = nn.Parameter(torch.Tensor([args.gamma]), requires_grad=False)
        self.neg = "T"   # toy   caser
        self.mask = None
        self.sampler = LogicalSampler(args, distribution="uniform")    # distribution="uniform" or "popularity"

        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        self.projection_regularizer = Regularizer(0.05, 0.05, 1e9)  # 0.05 for other
        self.fea2log = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


    def forward_sasrec(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        self.mask = item_seq != 0

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]


    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


    def feature_to_beta(self, feature):
        logic_input = self.fea2log(feature)
        logic_input = self.projection_regularizer(logic_input)
        alpha, beta = torch.chunk(logic_input, 2, dim=-1)

        return alpha, beta


    def vec_to_dis(self, alpha, beta):
        dis = torch.distributions.beta.Beta(alpha, beta)
        return dis


    def distance(self, dis1, dis2):
        score = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(dis1, dis2), p=1, dim=-1)
        return score


    def calculate_loss(self, seq, pos, neg, item_seq_len):
        # choose base model
        
        pos = pos.squeeze(-1)  # (B)
        neg = neg.squeeze(-1)  # (B)
        
        seq_output = self.forward_sasrec(seq, item_seq_len)

        neg_items_all = self.sampler.sample_neg_sequence(seq, self.neg_sam_num)
        #  feature2logic
        alpha_seq, beta_seq = self.feature_to_beta(self.item_embedding(seq))
        alpha_pos, beta_pos = self.feature_to_beta(self.item_embedding(pos))
        alpha_neg1, beta_neg1 = self.feature_to_beta(self.item_embedding(neg))
        
        alpha_neg2, beta_neg2 = self.feature_to_beta(self.item_embedding(neg_items_all))
        alpha_neg, beta_neg = self.Negation(alpha_neg2).view(alpha_seq.size(0), -1, self.hidden_size), \
                              self.Negation(beta_neg2).view(alpha_seq.size(0), -1, self.hidden_size)
        
        # logic input
        if self.neg == 'T':
            alpha_input, beta_input = torch.cat([alpha_neg, alpha_seq], dim=1), torch.cat([beta_neg, beta_seq], dim=1)
            self.mask = torch.cat([torch.ones(alpha_seq.size(0), alpha_neg.size(1)).cuda(), self.mask], dim=1).bool()
        else:
            alpha_input, beta_input = alpha_seq, beta_seq
            
        alpha_output, beta_output = self.logic_forward(alpha_input, beta_input)  # (B, E)
        out_dis = self.vec_to_dis(alpha_output, beta_output)
        pos_dis = self.vec_to_dis(alpha_pos, beta_pos)
        neg_dis = self.vec_to_dis(alpha_neg1, beta_neg1)

        logic_pos_score = self.distance(pos_dis, out_dis)
        logic_neg_score = self.distance(neg_dis, out_dis)
        logic_loss = self.loss_fct(logic_pos_score, logic_neg_score)

        if self.rec_loss_type == 'CE':
            #  prediction score
            test_item_emb = self.item_embedding.weight
            #  sample from Beta distribution
            logic_output = alpha_output / (alpha_output + beta_output)
            all_i_alpha, all_i_beta = self.feature_to_beta(test_item_emb)
            all_i_output = all_i_alpha / (all_i_alpha + all_i_beta)

            logits = torch.matmul(torch.cat([seq_output, logic_output], dim=1),
                                torch.cat([test_item_emb, all_i_output], dim=1).transpose(0, 1))

            loss1 = self.logic_CE(logits, pos)
        elif self.rec_loss_type == 'BCE':
            pos_emb = self.item_embedding(pos)  # (B, E)
            neg_emb = self.item_embedding(neg)  # (B, E) 
            logic_output = alpha_output / (alpha_output + beta_output)  # (B, E)
            pos_alpha, pos_beta = alpha_pos, beta_pos
            neg_alpha, neg_beta = alpha_neg1, beta_neg1
            pos_output = pos_alpha / (pos_alpha + pos_beta)
            neg_output = neg_alpha / (neg_alpha + neg_beta)
            seq_logic_output = torch.cat((seq_output, logic_output), dim=-1)  # (B, 2E)
            pos_logic_output = torch.cat((pos_emb, pos_output), dim=-1)  # (B, 2E)
            neg_logic_output = torch.cat((neg_emb, neg_output), dim=-1)  # (B, 2E)
            
            pos_logits = torch.sum(seq_logic_output * pos_logic_output, dim=-1).squeeze()  # (B)
            neg_logits = torch.sum(seq_logic_output * neg_logic_output, dim=-1).squeeze()  # (B)
            
            pos_label = torch.ones_like(pos_logits).cuda()
            neg_label = torch.zeros_like(neg_logits).cuda()
            
            loss1 = self.logic_BCE(pos_logits, pos_label)
            loss1 += self.logic_BCE(neg_logits, neg_label) 
            
        if self.base_model == 'Caser':
            reg_loss = self.reg_loss(
                [
                    self.user_embedding.weight,
                    self.item_embedding.weight,
                    self.conv_v.weight,
                    self.fc1.weight,
                    self.fc2.weight,
                    self.intersection.feature_layer_1.weight,
                    self.fea2log.weight
                ]
            )
        else:
            reg_loss = self.reg_loss(
                [
                    self.item_embedding.weight,
                    self.intersection.feature_layer_1.weight,
                    self.fea2log.weight
                ]
            )
              
        loss = loss1 + self.bpr_weight * logic_loss + self.reg_weight * reg_loss  # 0.5

        return loss


    def logic_forward(self, alphas, betas):

        alpha, beta = self.intersection(alphas, betas, self.mask)

        return alpha, beta


    def predict(self, seq, test_item, item_seq_len):
        seq_output = self.forward_sasrec(seq, item_seq_len)  # (B, E)

        alpha_seq, beta_seq = self.feature_to_beta(self.item_embedding(seq))
        alpha_output, beta_output = self.logic_forward(alpha_seq, beta_seq)
        logic_output = alpha_output / (alpha_output + beta_output)

        test_item_emb = self.item_embedding(test_item)
        
        all_i_alpha, all_i_beta = self.feature_to_beta(test_item_emb)
        all_i_output = all_i_alpha / (all_i_alpha + all_i_beta)  # (B, I, E)
        
        left = torch.cat([seq_output, logic_output], dim=-1).unsqueeze(1)  # (B, 1, 2E)
        right = torch.cat([test_item_emb, all_i_output], dim=-1)  # (B, I, 2E)
        
        scores = torch.matmul(left, right.transpose(-2, -1)).squeeze(1)  # (B, I)        
        # scores = torch.matmul(torch.cat([seq_output, logic_output], dim=1).unsqueeze(1),
        #                       torch.cat([test_item_emb, all_i_output], dim=1).transpose(-2, -1)).squeeze()  # (B, I)

        return scores


    def predict_full(self, seq, item_seq_len):
        seq_output = self.forward_sasrec(seq, item_seq_len)

        alpha_seq, beta_seq = self.feature_to_beta(self.item_embedding(seq))
        alpha_output, beta_output = self.logic_forward(alpha_seq, beta_seq)
        logic_output = alpha_output / (alpha_output + beta_output)

        test_items_emb = self.item_embedding.weight

        all_i_alpha, all_i_beta = self.feature_to_beta(test_items_emb)
        all_i_output = all_i_alpha / (all_i_alpha + all_i_beta)

        scores = torch.matmul(torch.cat([seq_output, logic_output], dim=1),
                              torch.cat([test_items_emb, all_i_output], dim=1).transpose(0, 1))

        # scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    
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
            # break
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
            

    



