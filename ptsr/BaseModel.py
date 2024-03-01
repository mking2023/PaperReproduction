import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptsr.datasets import *
from ptsr.utils import *


class Regularizer:
    def __init__(self, base_add, min_val, max_val) -> None:
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, item_embedding):
        return torch.clamp(item_embedding + self.base_add, self.min_val, self.max_val)


class BetaIntersection(nn.Module):
    def __init__(self, args) -> None:
        super(BetaIntersection, self).__init__()
        self.args = args
        self.emb_dim = args.emb_dim
        self.layer1 = nn.Linear(2 * self.emb_dim,  2 * self.emb_dim)
        self.layer2 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        
    def forward(self, alpha, beta, mask):
        """
        Args:
            alpha : ..., W, E   (W is window size, E is embedding dim)
            beta : ..., W, E
            mask : ..., W
        """
        all_embeddings = torch.cat([alpha, beta], dim=-1)  # ..., W, 2E
        mask = torch.where(mask, 0., -10000.)
        
        layer1_act = F.relu(self.layer1(all_embeddings))  # ..., W, 2E
        # attention_input = self.layer2(layer1_act) * mask.unsqueeze(-1) * self.args.tau  # ..., W, E
        attention_input = self.layer2(layer1_act)
        attention_input = attention_input + mask.unsqueeze(-1)
        attention = F.softmax(attention_input, dim=-2)  # ..., W, E
        alpha = torch.sum(alpha * attention, dim=-2)  # ..., E
        beta = torch.sum(beta * attention, dim=-2)  # ..., E
        return alpha, beta


class BetaProjection(nn.Module):
    def __init__(self, args, projection_regularizer) -> None:
        super(BetaProjection, self).__init__()
        self.args = args
        dim = args.dim
        self.layer1 = nn.Linear(args.max_len * dim * 2, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, dim)
        self.layer0 = nn.Linear(dim, dim)
        
        for nl in range(args.num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f'layer{nl}').weight)
        
        self.projection_regularizer = projection_regularizer
        
    def forward(self, seq_alpha, seq_beta, mask):
        """
        Args:
            seq_alpha : (B, S, E)
            seq_beta : (B, S, E)
            mask : (B, S)
        """
        x = torch.cat((seq_alpha, seq_beta), dim=-1)  # (B, S, 2 * E)
        x = x * mask.unsqueeze(-1)  # (B, S, 2 * E)
        for nl in range(1, self.args.num_layers + 1):
            x = F.relu(getattr(self, f'layer{nl}')(x))
        x = self.projection_regularizer(self.layer0(x))
        return x
            
    
class BetaNegation(nn.Module):
    def __init__(self) -> None:
        super(BetaNegation, self).__init__()
    
    def forward(self, embedding):
        embedding = 1. / embedding
        return embedding


class GammaIntersection(nn.Module):
    def __init__(self, args) -> None:
        super(GammaIntersection, self).__init__()
        dim = args.emb_dim
        self.layer_alpha1 = nn.Linear(dim * 2, dim)
        self.layer_alpha2 = nn.Linear(dim, dim)
        self.layer_beta1 = nn.Linear(dim * 2, dim)
        self.layer_beta2 = nn.Linear(dim, dim)     
        
    def forward(self, alpha_emb, beta_emb, mask):
        all_emb = torch.cat((alpha_emb, beta_emb), dim=-1)  # (W, 2E)
        
        mask = torch.where(mask, 1., -10000.)
        layer1_alpha = F.relu(self.layer_alpha1(all_emb))
        attention1 = self.layer_alpha2(layer1_alpha) * mask.unsqueeze(-1)  # (W, E)
        attention1 = F.softmax(attention1, dim=-2)  # (W, E)
        
        layer1_beta = F.relu(self.layer_beta1(all_emb))
        attention2 = self.layer_beta2(layer1_beta) * mask.unsqueeze(-1)  # (W, E)
        attention2 = F.softmax(attention2, dim=-2)  # (W, E)
        
        alpha = torch.sum(alpha_emb * attention1, dim=-2)  # (E)
        beta = torch.sum(beta_emb * attention2, dim=-2)  # (E)
        
        return alpha, beta


class GammaUnion(nn.Module):
    def __init__(self, args) -> None:
        super(GammaUnion, self).__init__()
        dim = args.emb_dim
        self.layer_alpha1 = nn.Linear(dim * 2, dim)
        self.layer_alpha2 = nn.Linear(dim, dim // 2)
        self.layer_alpha3 = nn.Linear(dim // 2, dim)
        
        self.layer_beta1 = nn.Linear(dim * 2, dim)
        self.layer_beta2 = nn.Linear(dim, dim // 2)
        self.layer_beta3 = nn.Linear(dim // 2, dim)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, alpha_emb, beta_emb):
        """
            alpha: (B, N, E)
            beta: (B, N, E)
            mask: (B, N)
        """
        # alpha_emb, beta_emb =  alpha * mask.unsqueeze(-1), beta * mask.unsqueeze(-1)  # (B, N, E)
        all_emb = torch.cat((alpha_emb, beta_emb), dim=-1)  # (B, N, 2E)
        layer1_alpha = F.relu(self.layer_alpha1(all_emb))  
        layer2_alpha = F.relu(self.layer_alpha2(layer1_alpha))
        attention1 = F.softmax(self.dropout(self.layer_alpha3(layer2_alpha)), dim=1)  # (B, N, E)
        
        layer1_beta = F.relu(self.layer_beta1(all_emb))
        layer2_beta = F.relu(self.layer_beta2(layer1_beta))
        attention2 = F.softmax(self.dropout(self.layer_beta3(layer2_beta)), dim=1)
        
        k = alpha_emb * attention1
        o = 1 / (beta_emb * attention2)
        k_sum = torch.pow(torch.sum(k * o, dim=1), 2) / torch.sum(torch.pow(o, 2) * k, dim=1)
        o_sum = torch.sum(k * o, dim=1) / (k_sum * o.shape[1])
        # Welch–Satterthwaite equation
        
        alpha_emb = k_sum  # (B, E)
        beta_emb = o_sum  # (B, E)
        alpha_emb[torch.abs(alpha_emb) < 1e-4] = 1e-4
        beta_emb[torch.abs(beta_emb) < 1e-4] = 1e-4
        return alpha_emb, beta_emb
        
        
class GammaProjection(nn.Module):      
    def __init__(self, args) -> None:
        super(GammaProjection, self).__init__()
        self.args = args
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.projection_hidden_dim
        
        self.layer_alpha1 = nn.Linear(self.emb_dim, self.hidden_dim)
        self.layer_alpha0 = nn.Linear(self.hidden_dim, self.emb_dim)
                    
        self.layer_beta1 = nn.Linear(self.emb_dim, self.hidden_dim)
        self.layer_beta0 = nn.Linear(self.hidden_dim, self.emb_dim)

        self.projection_regularizer = Regularizer(1, 0.15, 1e9)      
    
    def forward(self, alpha, beta):
        """
            alpha: (B, X, E)
            beta: (B, X, E)
        """
        all_alpha = alpha  # (B, X, E)
        all_beta = beta  # (B, X, E)
    
        all_alpha = F.relu(self.layer_alpha1(all_alpha))
        all_alpha = self.layer_alpha0(all_alpha)
        all_alpha = self.projection_regularizer(all_alpha)  # (B, X, E)
        
        all_beta = F.relu(self.layer_beta1(all_beta))
        all_beta = self.layer_beta0(all_beta)
        all_beta = self.projection_regularizer(all_beta)  # (B, X, E)
        
        return all_alpha, all_beta
    

class BaseModel(nn.Module):
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument('--emb_dim', type=int, default=64, help='dim of Embedding')
        parser.add_argument('--emb_type', type=str, default='gamma', help='general, beta, gamma')
        parser.add_argument('--base_add', type=float, default=1, help='make sure the parameters of Beta embedding is positive')
        parser.add_argument('--min_val', type=float, default=0.05, help='make sure the parameters of Beta embedding is positive')
        parser.add_argument('--max_val', type=float, default=1e9, help='make sure the parameters of Beta embedding is positive')
        parser.add_argument('--gamma', type=float, default=2., help='use to initialize embedding')
        parser.add_argument('--pattern_level', type=int, default=2, help='maximum value of sliding window')
        parser.add_argument('--mlp_lambda', type=float, default=1.0)
    
    
    def __init__(self, args) -> None:
        super(BaseModel, self).__init__()
        self.args = args
        
        emb_type = args.emb_type.lower()
        if emb_type in ['beta', 'gamma']:
            self.item_embedding = nn.Parameter(torch.zeros(self.args.item_num + 1, self.args.emb_dim * 2))  # (alpha, beta)
            self.gamma = nn.Parameter(torch.tensor([self.args.gamma]), requires_grad=False)
            
            if emb_type == 'beta': 
                self.regularizer = Regularizer(1, 0.05, 1e9) 
                self.embedding_range = nn.Parameter(torch.tensor([self.gamma.item() / self.args.emb_dim]), requires_grad=False)
                self.intersection = BetaIntersection(self.args)
                nn.init.uniform_(tensor=self.item_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            elif emb_type == 'gamma':
                self.epsilon = 2.0  # gamma embedding
                self.regularizer = Regularizer(1, 0.15, 1e9)
                self.embedding_range = nn.Parameter(torch.tensor([(self.gamma.item() + self.epsilon) / self.args.emb_dim]), requires_grad=False)
                self.intersection = GammaIntersection(self.args)                
                nn.init.uniform_(tensor=self.item_embedding, a=-3. * self.embedding_range.item(), b=3. * self.embedding_range.item())

    
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)                
    
    
    def vec_to_distribution(self, alpha, beta):
        if self.args.emb_type == 'beta':
            return torch.distributions.beta.Beta(alpha, beta)
        elif self.args.emb_type == 'gamma':
            return torch.distributions.gamma.Gamma(alpha, beta)
        else:
            raise ValueError("Error embedding type => {}".format(self.args.emb_type.lower()))
    
    
    def cal_distance(self, dist1, dist2):
        return self.gamma - torch.norm(torch.distributions.kl.kl_divergence(dist1, dist2), p=1, dim=-1)
    
    
    def get_embedding(self, indices):
        """beta or gamma"""
        emb = self.item_embedding[indices.long()]  # (..., 2E)
        emb = self.regularizer(emb)
        alpha, beta = torch.chunk(emb, 2, dim=-1)
        return alpha, beta
    
    
    def get_pattern_index(self, seq, window_size):
        """
        Args:
            seq : B, S
            position: B, S
            window_size : X
        """
        B, S = seq.shape[0], seq.shape[1]
        seq = seq.unsqueeze(0).unsqueeze(0).float()  # (1, 1, B, S)   input need to be 4D
        unfold = nn.Unfold(kernel_size=(1, window_size), stride=(1, 1))
        sub_seq = unfold(seq).transpose(-2, -1).reshape(B, -1, window_size)  # (B, S - W + 1, W) 
        mask = sub_seq != 0
        return sub_seq, mask 
            
    
    def cal_pattern_target_distance(self, w_alpha, w_beta, t_alpha, t_beta):
        """
            Args:
                w_alpha : B, X, E
                w_beta : B, X, E
                t_alpha : B, Y, E
                t_beta : B, Y, E
        """
        w_alpha = w_alpha.unsqueeze(1).repeat(1, t_alpha.shape[1], 1, 1)  # (B, Y, X, E)
        w_beta = w_beta.unsqueeze(1).repeat(1, t_alpha.shape[1], 1, 1)  # (B, Y, X, E)
        t_alpha = t_alpha.unsqueeze(2)  # (B, Y, 1, E)
        t_beta = t_beta.unsqueeze(2)  # (B, Y, 1, E)

        w_dist = self.vec_to_distribution(w_alpha, w_beta)
        t_dist = self.vec_to_distribution(t_alpha, t_beta)

        distance = self.cal_distance(t_dist, w_dist)
        
        return distance

    
    def forward(self, seq, pos, neg):
        raise NotImplementedError()
    
    
    def predict(self, seq, pos, neg):
        raise NotImplementedError()
    
    
    def calculate_loss(self, seq, pos, neg):
        raise NotImplementedError()
    

    def train_model(self, dataset):
        user_train, user_valid, user_test, user_num, item_num = dataset
        train_data = SequentialDataset(self.args, user_train, padding_mode='Left')
        valid_data = EvalDataset(self.args, user_train, user_valid, user_test, mode='valid', padding_mode='left')
        test_data = EvalDataset(self.args, user_train, user_valid, user_test, mode='test', padding_mode='left')
        
        train_loader = DataLoader(train_data, self.args.batchsize, shuffle=True)
        valid_loader = DataLoader(valid_data, self.args.batchsize_eval)
        test_loader = DataLoader(test_data, self.args.batchsize_eval) 
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_decay_step, gamma=self.args.lr_decay_rate)
                
        best_valid_bench = -1
        final_test_bench = -1.
        final_test_bench_string = None
        
        for epoch in range(1, self.args.epoch + 1):
            epoch_loss = self.train_an_epoch(train_loader, optimizer, scheduler)    
            out_str = "Epoch [{}] Loss = {:.5f} lr = {:.5f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr'])
            print(out_str)
            logging.info(out_str)
            if self.writer != None:
                self.writer.add_scalar('Loss', epoch_loss, epoch)
            
            if epoch % self.args.eval_interval == 0:
                valid_metric, valid_bench = self.evaluate_model(epoch, valid_loader, mode='Valid')
                test_metric, test_bench = self.evaluate_model(epoch, test_loader, mode='Test')
                
                print(valid_metric)
                print(test_metric)
                logging.info(valid_metric)
                logging.info(test_metric)              
                
                if valid_bench > best_valid_bench:
                    best_valid_bench = valid_bench
                    final_test_bench = test_bench
                    final_test_bench_string = test_metric
                    torch.save(self.state_dict(), os.path.join(self.args.log_dir, f'gamma_{self.args.dataset}_epoch_{epoch}.pth'))
        
        print("=" * 80, '\n', final_test_bench_string)
        logging.info("=" * 100)
        logging.info(final_test_bench_string)
        
        return best_valid_bench, final_test_bench
        
        
    def train_an_epoch(self, train_loader, optimizer, scheduler):
        epoch_loss = []
        self.train()
        for seq, pos, neg in add_process_bar(train_loader, desc='Train'):
            optimizer.zero_grad()
            seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
            loss = self.calculate_loss(seq, pos, neg)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        scheduler.step()  # lr decay
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
                scores, _ = self.predict(seq, pos, neg)   # (B, Y)
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
    
    
    def visualize_weight(self, dataset, save_path=None):
        user_train, user_valid, user_test, user_num, item_num = dataset
        test_data = EvalDataset(self.args, user_train, user_valid, user_test, mode='test', padding_mode='left')
        test_loader = DataLoader(test_data, self.args.batchsize_eval, pin_memory=True) 
        
        metric_name = ['NDCG', 'HR']
        topks = [5, 10]
        metric_obj = Metrics(metric_name, topks, is_full=False, mode='Test', benchmark_name='NDCG', benchmark_k=10)
        
        all_uid_lst = []
        all_seq_pos_lst = []
        all_pos_weight_lst = [] 
        all_pos_dis_lst = []
        all_pos_weight_dis_lst = []
        all_score_lst = []
        all_sample_lst = []
        all_neg_dis_lst = []
        all_neg_id_lst = []
        
        with torch.no_grad():
            self.eval()
            sample_num = 0
            cnt = 0
            for user, seq, pos, neg in add_process_bar(test_loader, desc='Test'):
                user, seq, pos, neg = user.cuda(), seq.cuda(), pos.cuda(), neg.cuda()
                # scores, pattern_weight = self.predict(seq, pos, neg)   # (B, Y)  （B, 1, N）
                scores, pos_weight, pos_dis, pos_weight_dis, neg_dis = self.case_study(seq, pos, neg)  # (B, Y)  (B, 1, N)
                sample_num += seq.shape[0]
                metric_obj.metric_value_accumulate(scores)
            
                all_uid_lst.append(user.unsqueeze(-1))
                all_seq_pos_lst.append(torch.cat((seq, pos), dim=-1))
                all_pos_weight_lst.append(pos_weight.squeeze(1))
                all_pos_dis_lst.append(pos_dis.squeeze(1))
                all_pos_weight_dis_lst.append(pos_weight_dis.squeeze(1))
                all_score_lst.append(scores)  # (B, Y)
                all_sample_lst.append(torch.cat((pos, neg), dim=-1))  # (B, Y)
                all_neg_dis_lst.append(neg_dis.squeeze(1))  # (B, N)
                all_neg_id_lst.append(neg[:, 3].unsqueeze(-1))  
                # cnt += 1
                # if cnt > 2:
                #     break
                
            all_user = torch.cat(all_uid_lst, dim=0)
            all_seq_pos = torch.cat(all_seq_pos_lst, dim=0)
            all_pos_weight = torch.cat(all_pos_weight_lst, dim=0)
            all_pos_dis = torch.cat(all_pos_dis_lst, dim=0)
            all_pos_weight_dis = torch.cat(all_pos_weight_dis_lst, dim=0)
            all_score = torch.cat(all_score_lst, dim=0)
            all_sample = torch.cat(all_sample_lst, dim=0)
            all_neg_dis = torch.cat(all_neg_dis_lst, dim=0)  
            all_neg_id = torch.cat(all_neg_id_lst, dim=0)
            
            print(all_user.shape)
            print(all_seq_pos.shape)
            print(all_pos_weight.shape)
            print(all_pos_dis.shape)
            print(all_pos_weight_dis.shape)
            print(all_score.shape)
            print(all_sample.shape)
            print(all_neg_dis.shape)
            print(all_neg_id.shape)
            
            
            all_user_seq = torch.cat((all_user, all_seq_pos), dim=-1).cpu().numpy()
            all_user_weight = torch.cat((all_user, all_pos_weight), dim=-1).cpu().numpy()
            all_user_dis = torch.cat((all_user, all_pos_dis), dim=-1).cpu().numpy()
            all_user_weight_dis = torch.cat((all_user, all_pos_weight_dis), dim=-1).cpu().numpy()
            all_user_score = torch.cat((all_user, all_score), dim=-1).cpu().numpy()
            all_user_sample = torch.cat((all_user, all_sample), dim=-1).cpu().numpy()
            all_user_neg_dis = torch.cat((all_user, all_neg_dis), dim=-1).cpu().numpy()
            all_user_neg_id = torch.cat((all_user, all_neg_id), dim=-1).cpu().numpy()
            
            
            print("all user seq = ", all_user_seq.shape)
            print("all user weigth = ", all_user_weight.shape)
            print("all user dis = ", all_user_dis.shape)
            print("all user weight dis = ", all_user_weight_dis.shape)
            print("all user score = ", all_user_score.shape)
            print("all user sample = ", all_user_sample.shape)
            print("all neg dis = ", all_user_neg_dis.shape)
            print("all_user neg id = ", all_user_neg_id.shape)
            
            start_time = time.time()
            user_seq_path = os.path.join(save_path, './user_seq.txt')
            user_weight_path = os.path.join(save_path, './user_weight.txt')
            user_dis_path = os.path.join(save_path, './user_dis.txt')
            user_weight_dis_path = os.path.join(save_path, './user_weight_dis.txt')
            user_score_path = os.path.join(save_path, './user_score.txt')
            user_sample_path = os.path.join(save_path, './user_sample.txt')
            user_neg_dis_path = os.path.join(save_path, './user_neg_dis.txt')
            user_neg_id_path = os.path.join(save_path, './user_neg_id.txt')
            
            # np.savetxt(user_seq_path, all_user_seq, fmt='%d', delimiter='\t')
            # np.savetxt(user_weight_path, all_user_weight, fmt='%.4f', delimiter='\t')
            # np.savetxt(user_dis_path, all_user_dis, fmt='%.4f', delimiter='\t')
            # np.savetxt(user_weight_dis_path, all_user_weight_dis, fmt='%.4f', delimiter='\t')
            # np.savetxt(user_score_path, all_user_score, fmt='%.4f', delimiter='\t')
            # np.savetxt(user_sample_path, all_user_sample, fmt='%.4f', delimiter='\t')
            np.savetxt(user_neg_dis_path, all_user_neg_dis, fmt='%.4f', delimiter='\t')
            np.savetxt(user_neg_id_path, all_user_neg_id, fmt='%.4f', delimiter='\t')
            end_time = time.time()
            print("Save Time Consume: [{:.2f}]".format(end_time - start_time))
            
            # metric_obj.average_metric_value(sample_num)
            # metric_string = metric_obj.metric_value_to_string()
            # benchmark_value = metric_obj.get_benchmark_value()
            # print(metric_string)

        
            
                
        
        
        
        
    
        