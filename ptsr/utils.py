import os
import time
import logging
import random

import wandb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from collections import defaultdict
from pprint import PrettyPrinter
from multiprocessing import Process, Queue
from tqdm import tqdm
from functools import wraps
from datetime import datetime


def data_partition(filepath, user_field, item_field, time_field, sep='\t'):
    df = pd.read_csv(filepath, sep=sep, header=0)[[user_field, item_field, time_field]]
    df = df.sort_values(by=[user_field, time_field])

    User = defaultdict(list)
    user_num, item_num = 0, 0
    user_train, user_valid, user_test = {}, {}, {}

    for i, row in df.iterrows():
        u = int(row[user_field])
        i = int(row[item_field])
        user_num = max(user_num, u)
        item_num = max(item_num, i)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        user_valid[user] = []
        user_test[user] = []
        if nfeedback < 3:
            user_train[user] = User[user]
        else:
            user_train[user] = User[user][:-2]
            user_valid[user].append(User[user][-2])
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, user_num, item_num]
    
    
def cal_mrr_accum(preds, k, pos=None):
    if pos is None:
        preds_rank = (-preds).argsort().argsort()[:, 0]  # (B)
    else:
        preds_rank = (-preds).argsort().argsort()  # (B, N)
        preds_rank = torch.gather(preds_rank, -1, pos)
    topkid = preds_rank < k
    mrr_k = torch.reciprocal(preds_rank[topkid].float() + 1).sum()
    return mrr_k


def cal_hr_accum(preds, k, pos=None):
    if pos is None:
        preds_rank = (-preds).argsort().argsort()[:, 0]  # (B)
    else:
        preds_rank = (-preds).argsort().argsort()  # (B, N)
        preds_rank = torch.gather(preds_rank, -1, pos)

    topkid = preds_rank < k
    hr_k = topkid.sum().item()
    return hr_k


def cal_ndcg_accum(preds, k, pos=None):
    if pos is None:
        preds_rank = (-preds).argsort().argsort()[:, 0]  # (B)
    else:
        preds_rank = (-preds).argsort().argsort()  # (B, N)
        preds_rank = torch.gather(preds_rank, -1, pos)
    topkid = preds_rank < k
    valid_rank = preds_rank[topkid]
    ndcg_k = 0.0
    for rank in valid_rank.cpu().numpy():
        ndcg_k += 1. / np.log2(rank + 2)
    return ndcg_k


def print_args(arg):
    pp = PrettyPrinter(indent=4)
    pp.pprint(vars(arg))
    log_args(arg)

    
def ensure_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
            

def init_log(args):
    timestamp = time.strftime("%m%d_%H_%M_%S", time.localtime())
    log_dir_model = os.path.join('./log', args.model)
    log_dir_dataset = os.path.join(log_dir_model, args.dataset)
    log_dir_exp = os.path.join(log_dir_dataset, timestamp)
    ensure_dir(log_dir_model)
    ensure_dir(log_dir_dataset)
    ensure_dir(log_dir_exp)
    
    logging.basicConfig(
        filename=os.path.join(log_dir_exp, 'train_log.txt'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return log_dir_exp


def log_args(args):
    arg_keys = []
    for k in vars(args):
        arg_keys.append(k)
    
    arg_keys = sorted(arg_keys, key=lambda x: len(x))
    for k in arg_keys:
        logging.info("{}\t-> {}".format(k, getattr(args, k)))
    

def init_environment(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(args.gpu)
    

def print_model_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total /= 1000000
    print("Total Parameters of [{}] is [{}] M".format(model.__class__.__name__, total))
    
    
def add_process_bar(data, desc=None):
    if desc is not None:
        data = tqdm(data, total=len(data), leave=False, ncols=70, desc=desc)
    else:
        data = tqdm(data, total=len(data), leave=False, ncols=70)
    return data



NameToFunction = {
    'NDCG' : cal_ndcg_accum,
    'HR' : cal_hr_accum,
    'MRR' : cal_mrr_accum,
}


class Metrics:
    def __init__(self, metric_name: list, topks: list, is_full=0, mode='Test', benchmark_name='NDCG', benchmark_k=10) -> None:
        self.metric_name = metric_name
        self.topks = topks
        self.mode = mode
        self.is_full = is_full
        self.benchmark_name = benchmark_name
        self.benchmark_k = benchmark_k
        self.metric_dict = {}
        self.init_metric_dict()

        assert benchmark_k in self.topks, ValueError(f'Error benchmark_k [{benchmark_k}]')
    
    
    def init_metric_dict(self):
        for name in self.metric_name:
            self.metric_dict[name] = {}
            for k in self.topks:
                self.metric_dict[name][k] = 0        
    
    
    def metric_value_accumulate(self, scores, target=None):
        if self.is_full:
            assert target is not None, ValueError('Target can not be None when [is_full==True]')
        else:
            target = None
        for name in self.metric_name:
            func = NameToFunction[name]
            for k in self.topks:
                value = func(scores, k, target)
                self.metric_dict[name][k] += value
    
    
    def average_metric_value(self, sample_num):
        for name in self.metric_name:
            for k in self.topks:
                self.metric_dict[name][k] /= float(sample_num)
        
    
    def metric_value_to_string(self):
        s = f'\t[{self.mode:>5}] >>> '
        is_first = True
        new_line_condition = (len(self.metric_name) * len(self.topks)) >= 6
        for i, name in enumerate(self.metric_name):
            for topk in self.topks:
                if is_first:
                    cur_s = '{:>4}{:>2} = {:.5f}'.format(name, topk, self.metric_dict[name][topk])
                    is_first = False
                else:
                    cur_s = ', {:>4}{:>2} = {:.5f}'.format(name, topk, self.metric_dict[name][topk])
                s += cur_s
            if new_line_condition and i < len(self.metric_name) - 1:
                is_first = True
                s += f'\n\t{">>>":>11} '
        return s        
    
    
    def get_benchmark_value(self):
        return self.metric_dict[self.benchmark_name][self.benchmark_k]


def get_model(args, parser):
    import baseline
    import ptsr
    if args.model in ['SASRec', 
                      'GRU4Rec',
                      'FMLPRecModel',
                      'DistSAModel',
                      'NextItNet',
                      'SRPLR']:
        class_obj = getattr(baseline, args.model)
        class_obj.parse_args(parser)
    elif args.model in ['FixedPattern', 
                        'GeneralEmb', 
                        'LastItem',
                        'FixedPatternWeight', 
                        'FixedPatternWeight2',
                        'FixedPatternPair', 
                        'FixedPatternAttention',
                        'PatternProjection',
                        'FixedPatternWeightMixer']:
        class_obj = getattr(ptsr, args.model)
        class_obj.parse_args(parser)
    else:
        raise ValueError('Error model name')
    args, _ = parser.parse_known_args()
    return args, class_obj


def RunTime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.fromtimestamp(time.time())
        result = func(*args, **kwargs)
        end_time = datetime.fromtimestamp(time.time())
        print("Total Time spent => [{}]".format(str(end_time - start_time)))
        return result
    return wrapper



class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

