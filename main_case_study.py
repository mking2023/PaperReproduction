import os
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import tuneparam

from baseline import *
from torch.utils.tensorboard import SummaryWriter


torch.set_printoptions(precision=7)
torch.set_printoptions(threshold=10000)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def init_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_step', type=int, default=1000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--batchsize_eval', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--neg_num', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Beauty')
    parser.add_argument('--model', type=str, default='BetaESR')  
    parser.add_argument('--tune_param', type=int, default=0)
    parser.add_argument('--full_sort', type=int, default=0)
    args, _ = parser.parse_known_args()
    
    init_environment(args)
        
    return args, parser
    

@RunTime
def main():
    args, parser = init_args()        
    dataset = data_partition(f'dataset/{args.dataset}/{args.dataset}.csv', 'uid', 'iid', 'time')  # train, valid, test, user_num, item_num
    print(f"user num = {dataset[3]},\nitem num = {dataset[4]}")

    if args.tune_param:
        func = getattr(tuneparam, f'{args.model}_tune')
        func(args, dataset, parser)
    else:
        args, model_class = get_model(args, parser)
        args.log_dir = init_log(args) 
        args.user_num = dataset[3] 
        args.item_num = dataset[4]
        print_args(args)
        writer = SummaryWriter(log_dir=args.log_dir)
        
        model = model_class(args, writer).cuda()
        print_model_parameters(model)
        model.train_model(dataset)
            

def load_model():
    # fixedPatternWeight_Tools_path = './log/FixedPatternWeight/Tools_meta/1229_19_53_53/gamma_Tools_meta_epoch_190.pth'
    # save_path = './log/FixedPatternWeight/Tools_meta/1229_19_53_53/'
    
    # fixedPatternWeight_mk100k_path = './log/FixedPatternWeight/ml-100k/1229_16_43_02/gamma_ml-100k_epoch_130.pth'
    # save_path = './log/FixedPatternWeight/ml-100k/1229_16_43_02'
    
    # fixedPatternWeight_Toys_path = './log/FixedPatternWeight/Toys/1229_20_23_01/gamma_Toys_epoch_80.pth'
    # save_path = './log/FixedPatternWeight/Toys/1229_20_23_01/'
    
    # fixedPatternWeight_Beauty_path = './log/FixedPatternWeight/Beauty/0107_13_43_52/gamma_Beauty_epoch_135.pth'
    # SASRec_Beauty_path = './log/SASRec/Beauty/0107_16_19_18/sasrec_Beauty_epoch_200.pth'
    
    fixedPatternWeightMixer_Beauty_path = './caseStudy/Beauty/gamma_Beauty_epoch_135.pth'
    SASRec_Beauty_path = './caseStudy/Beauty/SASRec/sasrec_Beauty_epoch_200.pth'
    
    args, parser = init_args()
    dataset = data_partition(f'dataset/{args.dataset}/{args.dataset}.csv', 'uid', 'iid', 'time')  # train, valid, test, user_num, item_num
        
    if args.model == 'FixedPatternWeightMixer':
        args, model_class = get_model(args, parser)
        args.user_num = dataset[3] 
        args.item_num = dataset[4]
        print_args(args)
        
        model = model_class(args, None).cuda()
        checkpoint = torch.load(fixedPatternWeightMixer_Beauty_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.cuda()
        model.visualize_weight(dataset, '/'.join(fixedPatternWeightMixer_Beauty_path.split('/')[:-1]))
    elif args.model == 'SASRec':
        args, model_class = get_model(args, parser)
        args.user_num = dataset[3] 
        args.item_num = dataset[4]
        print_args(args)
        
        model = model_class(args, None)
        checkpoint = torch.load(SASRec_Beauty_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.cuda()
        model.visualize_weight(dataset, '/'.join(SASRec_Beauty_path.split('/')[:-1]))


if __name__ == '__main__':
    # main()
    load_model()
    
