import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ptsr.utils import *


SASRec_hyper_params = {
    'lr': [1e-3, 5e-4, 1e-4],
    'dropout_rate': [0.5, 0.3, 0.2],  # [0.5, 0.3, 0.2] for ml-100k, [0.5, 0.3, 0.] for others
    'weight_decay': [0., 1e-8, 1e-7],
}


GRU4Rec_hyper_params = {
    'lr': [1e-3, 5e-4, 1e-4],
    'dropout_rate': [0.5, 0.3, 0.2, 0.0],
    'weight_decay': [0., 1e-8, 1e-7]
}



SRPLR_hyper_params = {
    'lr': [1e-3],
    'dropout_rate': [0.5, 0.3, 0.2],
    'bpr_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
    'reg_weight': [0.005],
}


Gamma_hyper_params = {
    'lr': [0.001],
    'weight_decay': [0., 1e-9, 1e-8, 1e-7],
    'mlp_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}



def SASRec_tune(args, dataset, parser):    
    specific_param = SASRec_hyper_params
    tune_time = time.strftime("%m%d_%H_%M_%S", time.localtime())
    args, model_class = get_model(args, parser)
    args.user_num = dataset[3] 
    args.item_num = dataset[4]
    cnt = 1
    for lr in specific_param['lr']:
        for dropout_rate in specific_param['dropout_rate']:
            for weight_decay in specific_param['weight_decay']:
                single_start_time = datetime.fromtimestamp(time.time())
                print("*" * 80)
                print(f"Exp Count = [{cnt}]")
                cnt += 1
                
                writer_dir = f'./log/{args.model}/{args.dataset}/tunehparam/{tune_time}/lr{lr}_drop{dropout_rate}_wd{weight_decay}'
                writer = SummaryWriter(writer_dir)

                setattr(args, 'log_dir', writer_dir)
                setattr(args, 'lr', lr)
                setattr(args, 'dropout_rate', dropout_rate)
                setattr(args, 'weight_decay', weight_decay)
                print_args(args)

                print('=' * 80)
                print("HyperParam => [lr: {}, dropout_rate: {}, weight_decay: {}]".format(lr, dropout_rate, weight_decay))
                print('=' * 80)                
                
                model = model_class(args, writer).cuda()
                print_model_parameters(model)
                best_valid_res, final_test_res = model.train_model(dataset)
            
                writer.add_hparams(
                    {
                        'lr': lr, 
                        'dropout_rate': dropout_rate, 
                        'weight_decay': weight_decay
                    },
                    {
                        'Valid_NDCG10': best_valid_res,
                        'Test_NDCG10': final_test_res
                    })                
                single_end_time = datetime.fromtimestamp(time.time())
                print("Time spent = [{}]".format(str(single_end_time - single_start_time)))
    writer.close()
    
    
def GRU4Rec_tune(args, dataset, parser):    
    specific_param = GRU4Rec_hyper_params
    tune_time = time.strftime("%m%d_%H_%M_%S", time.localtime())
    args, model_class = get_model(args, parser)
    args.user_num = dataset[3] 
    args.item_num = dataset[4]
    cnt = 1
    for lr in specific_param['lr']:
        for dropout_rate in specific_param['dropout_rate']:
            for weight_decay in specific_param['weight_decay']:
                single_start_time = datetime.fromtimestamp(time.time())
                print("*" * 80)
                print(f"Exp Count = [{cnt}]")
                cnt += 1
                
                writer_dir = f'./log/{args.model}/{args.dataset}/tunehparam/{tune_time}/lr{lr}_drop{dropout_rate}_wd{weight_decay}'
                writer = SummaryWriter(writer_dir)

                setattr(args, 'log_dir', writer_dir)
                setattr(args, 'lr', lr)
                setattr(args, 'dropout_rate', dropout_rate)
                setattr(args, 'weight_decay', weight_decay)
                print_args(args)
                
                print('=' * 80)
                print("HyperParam => [lr: {}, dropout_rate: {}, weight_decay: {}]".format(lr, dropout_rate, weight_decay))
                print('=' * 80)
                
                model = model_class(args, writer).cuda()
                print_model_parameters(model)
                best_valid_res, final_test_res = model.train_model(dataset)
            
                writer.add_hparams(
                    {
                        'lr': lr, 
                        'dropout_rate': dropout_rate, 
                        'weight_decay': weight_decay
                    },
                    {
                        'Valid_NDCG10': best_valid_res,
                        'Test_NDCG10': final_test_res
                    })             
                single_end_time = datetime.fromtimestamp(time.time())
                print("Time spent = [{}]".format(str(single_end_time - single_start_time)))
                   
    writer.close()


def SRPLR_tune(args, dataset, parser):    
    specific_param = SRPLR_hyper_params
    tune_time = time.strftime("%m%d_%H_%M_%S", time.localtime())
    args, model_class = get_model(args, parser)
    args.user_num = dataset[3] 
    args.item_num = dataset[4]
    cnt = 1
    for lr in specific_param['lr']:
        for attn_dropout_rate in specific_param['dropout_rate']:
            for bpr_weight in specific_param['bpr_weight']:
                for reg_weight in specific_param['reg_weight']:
                    single_start_time = datetime.fromtimestamp(time.time())
                    print("*" * 80)
                    print(f"Exp Count = [{cnt}]")
                    cnt += 1
                    
                    writer_dir = f'./log/{args.model}/{args.dataset}/tunehparam/{tune_time}/lr{lr}_drop{attn_dropout_rate}_bprw{bpr_weight}_regw{reg_weight}'
                    writer = SummaryWriter(writer_dir)

                    setattr(args, 'log_dir', writer_dir)
                    setattr(args, 'lr', lr)
                    setattr(args, 'attn_dropout_rate', attn_dropout_rate)
                    setattr(args, 'bpr_weight', bpr_weight)
                    setattr(args, 'reg_weight', reg_weight) 
                    print_args(args)
                    
                    print('=' * 80)
                    print("HyperParam => [lr: {}, attn_dropout_rate: {}, bpr_weight: {}, reg_weight: {}]".format(lr, attn_dropout_rate, bpr_weight, reg_weight))
                    print('=' * 80)
                    
                    model = model_class(args, writer).cuda()
                    print_model_parameters(model)
                    best_valid_res, final_test_res = model.train_model(dataset)
                
                    writer.add_hparams(
                        {
                            'lr': lr, 
                            'attn_dropout_rate': attn_dropout_rate, 
                            'bpr_weight': bpr_weight,
                            'reg_weight': reg_weight,
                        },
                        {
                            'Valid_NDCG10': best_valid_res,
                            'Test_NDCG10': final_test_res
                        })             
                    single_end_time = datetime.fromtimestamp(time.time())
                    print("Time spent = [{}]".format(str(single_end_time - single_start_time)))
                   
    writer.close()
    
    
def FixedPatternWeightMixer_tune(args, dataset, parser):
    specific_param = Gamma_hyper_params
    tune_time = time.strftime("%m%d_%H_%M_%S", time.localtime())
    args, model_class = get_model(args, parser)
    args.user_num = dataset[3] 
    args.item_num = dataset[4]
    cnt = 1
    
    random_seed_lst = [42, 2013, 2023, 2033, 3407]
    for lr in specific_param['lr']:
        for weight_decay in specific_param['weight_decay']:
            for mlp_lambda in specific_param['mlp_lambda']:
                single_start_time = datetime.fromtimestamp(time.time())
                print("*" * 80)
                print(f"Exp Count = [{cnt}]")
                cnt += 1
                
                writer_dir = f'./log/{args.model}/{args.dataset}/tunehparam/{tune_time}/lr{lr}_wd{weight_decay}_mlp{mlp_lambda}'
                writer = SummaryWriter(writer_dir)

                setattr(args, 'log_dir', writer_dir)
                setattr(args, 'lr', lr)
                setattr(args, 'weight_decay', weight_decay)
                setattr(args, 'mlp_lambda', mlp_lambda)
                print_args(args)
                
                print('=' * 80)
                print("HyperParam => [lr: {}, weight_decay: {}, mlp_lambda: {}]".format(lr, weight_decay, mlp_lambda))
                print('=' * 80)
                
                model = model_class(args, writer).cuda()
                print_model_parameters(model)
                best_valid_res, final_test_res = model.train_model(dataset)
            
                writer.add_hparams(
                    {
                        'lr': lr, 
                        'weight_decay': weight_decay,
                        'mlp_lambda': mlp_lambda
                    },
                    {
                        'Valid_NDCG10': best_valid_res,
                        'Test_NDCG10': final_test_res
                    })             
                single_end_time = datetime.fromtimestamp(time.time())
                print("Time spent = [{}]".format(str(single_end_time - single_start_time)))
                   
    writer.close()