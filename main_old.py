import os
import time
import wandb
import argparse
import logging
import pprint
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import ptsr
import baseline
from baseline import *
from ptsr import model_auto_pattern
from ptsr import model_fixed_pattern
from collections import defaultdict
from ptsr.utils import *
from ptsr.datasets import *
from tqdm import tqdm
from pprint import PrettyPrinter

torch.set_printoptions(precision=10)


USE_WANDB = False
TUNE_PARAM = False
LOAD_PRETRAIN = False
project_name = "BetaESR"


def init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--batchsize_eval', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--iter_interval', type=int, default=5)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Beauty')
    parser.add_argument('--model', type=str, default='BetaESR')  
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    
    args, _ = parser.parse_known_args()
        
    timestamp = time.strftime("%m%d_%H_%M_%S", time.localtime())
    log_dir_model = os.path.join('./log', args.model)
    log_dir_exp = os.path.join(log_dir_model, timestamp)
    ensure_dir(log_dir_model)
    ensure_dir(log_dir_exp)
    args.log_dir = log_dir_exp
    
    logging.basicConfig(
        filename=os.path.join(log_dir_exp, 'train_log.txt'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    torch.cuda.set_device(args.gpu)
    # init_seed(args.seed)    
    return args, parser

args = init()
    

def get_data(args):
    data_path = 'dataset/{}/{}.csv'.format(args.dataset, args.dataset)
    user_train, user_valid, user_test, u_num, i_num = data_partition(data_path, 'uid', 'iid','time')
    print(f"user num = {u_num}\nitem num = {i_num}")
    
    train_data = SequentialDataset(user_train, max_len=args.max_len, item_num=i_num)
    valid_data = EvalDataset(args.max_len, i_num, user_train, user_valid, user_test, mode='Valid')
    test_data = EvalDataset(args.max_len, i_num, user_train, user_valid, user_test, mode='Test')
    
    train_loader = DataLoader(train_data, args.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_data, args.batchsize_eval)
    test_loader = DataLoader(test_data, args.batchsize_eval)
    return train_loader, valid_loader, test_loader, i_num


def load_model(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model
    

def train_an_epoch(args, model, epoch, trainloader, optimizer, count, scheduler):
    epoch_loss = []
    model.train()
    for seq, pos, neg in tqdm(trainloader, total=len(trainloader), desc=f'Epoch [{epoch}]', ncols=100, leave=False):
        optimizer.zero_grad()
        seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
        loss = model.cal_loss(seq, pos, neg)
        
        if count % args.iter_interval == 0 and USE_WANDB:
            wandb.log({"loss_iter": loss.item()})
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        count += 1
    scheduler.step()
    
    return model, count, np.mean(epoch_loss)


def evaluate(args, model, dataloader, mode='Valid'):
    NDCG10, HR10 = 0., 0.
    with torch.no_grad():
        model.eval()
        sample_num = 0
        for seq, pos, neg in tqdm(dataloader, total=len(dataloader), desc=f'{mode}', ncols=100, leave=False):
            seq, pos, neg = seq.cuda(), pos.cuda(), neg.cuda()
            scores = model.predict(seq, pos, neg)  # (B, Y)
            NDCG10 += cal_ndcg_accum(scores, k=10)
            HR10 += cal_hr_accum(scores, k=10)
            sample_num += seq.shape[0]
        NDCG10 /= sample_num
        HR10 /= sample_num
        return NDCG10, HR10    

    
def betaesr_main(args):
    train_loader, valid_loader, test_loader, item_num = get_data(args)
    args.item_num = item_num
    if USE_WANDB:
        wandb.init(project=project_name, group='auto_pattern')
        if TUNE_PARAM:
            for k, v in wandb.config.items():
                setattr(args, k, v)
    print_args(args)
    logging.info("Hyper parameters")
    for k in vars(args):
        logging.info("{}\t={}".format(k, getattr(args, k)))
    
    model = Net(args).cuda()    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)     
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    cnt = 0 
    test_result = None
    best_v_ndcg = -1.
    for epoch in range(1, args.epoch + 1):
        model, cnt, epoch_loss = train_an_epoch(args, model, epoch, train_loader, optimizer, cnt, scheduler)
        out_str = "Epoch [{}], Loss = {:.5f}, lr = {:.7f}".format(epoch, epoch_loss, optimizer.param_groups[0]['lr'])
        print(out_str)
        logging.info(out_str)
        
        if USE_WANDB:
            wandb.log({"Loss": epoch_loss})
        if epoch % args.eval_interval == 0:
            valid_ndcg, valid_hr = evaluate(args, model, valid_loader, mode='Valid')
            test_ndcg, test_hr = evaluate(args, model, test_loader, mode='Test')
            if valid_ndcg > best_v_ndcg:
                best_v_ndcg = valid_ndcg
                test_result = (test_ndcg, test_hr)
                logging.info("Saving Model to {}".format(os.path.join(args.log_dir, f'best_model_epoch_{epoch}.pth')))
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'best_model_epoch_{epoch}.pth'))
            print("\tVali NDCG10 = {:.5f}, Vali HR10 = {:.5f}".format(valid_ndcg, valid_hr))
            print("\tTest NDCG10 = {:.5f}, Test HR10 = {:.5f}".format(test_ndcg, test_hr))
            logging.info("Vali NDCG10 = {:.5f},\tVali HR10 = {:.5f}".format(valid_ndcg, valid_hr))
            logging.info("Test NDCG10 = {:.5f},\tTest HR10 = {:.5f}".format(test_ndcg, test_hr))
    if USE_WANDB:
        wandb.log({"Test NDCG10": test_result[0], "Test HR10": test_result[1]})
    print("Test NDCG10 = {:.5f}, Test HR10 = {:.5f}".format(test_result[0], test_result[1]))
    logging.info("Test NDCG10 = {:.5f}, Test HR10 = {:.5f}".format(test_result[0], test_result[1]))
    

# def baseline_main():
#     args = init()
#     print_args(args)
#     data_path = 'dataset/{}/{}.csv'.format(args.dataset, args.dataset)
#     dataset = data_partition(data_path, 'uid', 'iid','time')  # train, valid, test, u_num, i_num
#     args.item_num = dataset[4]
    
#     if args.model.lower() == 'sasrec':
#         model = sasrec.SASRec(dataset[3], dataset[4], args).cuda()
#         sasrec.SASRec.training(args, model, dataset)
#     elif args.model.lower() == 'gru4rec':
#         model = gru4rec.GRU4Rec(args).cuda()
#         gru4rec.GRU4Rec.training(args, model, dataset)
    

def main():
    args, parser = init()
    # dataset = data_partition(f'dataset/{args.dataset}/{args.dataset}.csv', 'uid', 'iid', 'time')  # train, valid, test, user_num, item_num
    # print(f"user num = {dataset[3]},\titem num = {dataset[4]}")
    # args.item_num = dataset[4]
    # print_args(args)
    class_obj = getattr(baseline, 'SASRec')
    class_obj.parse_args(parser)
    
    args, _ = parser.parse_known_args()
    args.item_num = 10
    
    model = class_obj(args)

    for name, param in model.named_parameters():
        print(name, " => ", param.shape)

            

if __name__ == '__main__':
    main()
    # baseline_main()
    # betaesr_main()
    
    # if USE_WANDB:
    #     sweep_config = {
    #         "method": "grid",
    #         "name": "BetaESR_Beauty_2",
    #         "metric": {
    #             "goal": "minimize",
    #             "name": "Loss"
    #         },
    #         "parameters": {
    #             "epoch": {
    #                 "value": 20
    #             },
    #             "lr": {
    #                 "value": 0.001
    #             },
    #             "emb_dim": {
    #                 "values": [64, 128]
    #             },
    #             "gamma": {
    #                 "values": [1, 2, 3, 4, 5]
    #             },
    #             "tau": {
    #                 "value": 0.01
    #             },
    #             "pattern_level": {
    #                 "values": [4, 8]
    #             }
    #         }
    #     }

    # sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    # wandb.agent(sweep_id, function=train)
    



