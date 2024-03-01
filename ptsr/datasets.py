import time
import torch
import logging
import multiprocessing
import torch.nn as nn
import numpy as np

from tqdm import trange, tqdm
from multiprocessing import Process, Queue, Pool, cpu_count
from torch.utils.data import Dataset, DataLoader
from ptsr.utils import *


class SequentialDataset(Dataset):
    def __init__(self, args, user_train, padding_mode='left'):
        super(SequentialDataset, self).__init__()
        self.train_data = user_train
        self.max_len = args.max_len
        self.item_num = args.item_num
        self.padding_mode = padding_mode
        self.users = sorted(user_train.keys())
        self.all_train_data = []
        self.data_augmentation()  # 数据增强
        
        
    def data_augmentation(self):
        for i in self.users:
            self.all_train_data.extend(self.augment_single_user(i))
        logging.info("[Augmentation], Sample num is [{}]".format(len(self.all_train_data)))
        np.random.shuffle(self.all_train_data) 
        logging.info("[Shuffle] train data")
            
            
    def augment_single_user(self, uid):
        generated_seqs = []
        seq = self.train_data[uid]
        for i in range(2, len(seq) + 1):
            generated_seqs.append([uid] + seq[:i])  # [uid seq pos]
        return generated_seqs


    def __len__(self):
        return len(self.all_train_data)


    def __getitem__(self, index):
        user_seq = self.all_train_data[index]
        uid = user_seq[0]
        pos = [user_seq[-1]]
        seq = user_seq[1:-1]
        user_set = set(self.train_data[uid])
        
        t = np.random.randint(1, self.item_num + 1)
        while t in user_set:
            t = np.random.randint(1, self.item_num + 1)
        neg = [t]
        if self.padding_mode.lower() == 'left':
            seq = [0] * (self.max_len - len(seq)) + seq
            seq = seq[-self.max_len:]
            return torch.from_numpy(np.array(seq)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg))
        elif self.padding_mode.lower() == 'right':
            seq_len = [min(self.max_len, len(seq))]
            seq = seq + [0] * (self.max_len - len(seq))
            seq = seq[-self.max_len:]
            return torch.from_numpy(np.array(seq)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg)), torch.from_numpy(np.array(seq_len))
        else:
            raise ValueError('Error Mode in SequentialDataset __getitem__()')
    
    
class ParallelSequentialDataset(Dataset):
    def __init__(self, user_train: dict, max_len, item_num, shuffle=False) -> None:
        super(ParallelSequentialDataset, self).__init__()
        self.user_train = user_train
        self.max_len = max_len
        self.item_num = item_num
        self.user_lst = []
        for u in self.user_train:
            if len(self.user_train[u]) > 1:
                self.user_lst.append(u)
        self.user_lst = sorted(self.user_lst)
        if shuffle:
            np.random.shuffle(self.user_lst)
    
    
    def __len__(self):
        return len(self.user_lst)
    
    
    def __getitem__(self, index):
        uid = self.user_lst[index]
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        
        nxt = self.user_train[uid][-1]
        idx = self.max_len - 1
        
        ts = set(self.user_train[uid])
        for i in reversed(self.user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, self.item_num + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        uid, seq, pos, neg = np.array(uid), np.array(seq), np.array(pos), np.array(neg)
        uid, seq, pos, neg = torch.from_numpy(uid), torch.from_numpy(seq), torch.from_numpy(pos), torch.from_numpy(neg)    
        return uid, seq, pos, neg


class EvalDataset(Dataset):
    def __init__(self, args, user_train: dict, user_valid: dict, user_test: dict, mode='valid', padding_mode='left') -> None:
        super(EvalDataset, self).__init__()
        self.mode = mode
        self.max_len = args.max_len
        self.item_num = args.item_num
        self.neg_num = args.neg_num
        self.num_process = 6
        self.padding_mode = padding_mode
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.user_lst = self.get_user_lst()
        self.user_seq, self.user_pos, self.user_neg = self.get_data_of_all_user()
    
    
    def get_user_lst(self):
        user_lst = []
        for u in self.user_train.keys():
            if self.mode == 'valid' and len(self.user_valid[u]) >= 1:
                user_lst.append(u)
            elif self.mode == 'test' and len(self.user_test[u]) >= 1:
                user_lst.append(u)
        user_lst = sorted(user_lst)
        return user_lst
    
    
    def get_chunk_user_lst(self):
        user_chunks = []
        chunk_size = len(self.user_lst) // self.num_process
        for i in range(0, len(self.user_lst), chunk_size):
            user_chunks.append(self.user_lst[i: i + chunk_size])
        return user_chunks
    
    
    def get_data_of_chunk_user(self, user_chunk):
        chunk_results = []
        for u in user_chunk:
            uid, seq, pos, neg = self.get_data_of_one_user(u)
            chunk_results.append((uid, seq, pos, neg))
        return chunk_results
            
    
    def get_data_of_all_user(self):
        assert len(self.user_lst) > 0
        user_pos = {}
        user_seq = {}
        user_neg = {}
        
        print("=" * 80)
        print(f"Mode = [{self.mode}]")
        print("Multi-process negative sampling => process num = [{}]".format(self.num_process))
        start_time = time.time()
        user_chunks = self.get_chunk_user_lst()
        pool = Pool(processes=self.num_process)
        results = pool.map(self.get_data_of_chunk_user, user_chunks)
        all_results = []
        for r in results:
            all_results.extend(r)
        unique_users = set()
        end_time = time.time()
        print("Negative sampling completed => time: [{:.4f}] s".format(end_time - start_time))
        
        for row in add_process_bar(all_results, desc='Integrate Data'):
            uid, seq, pos, neg = row
            user_seq[uid] = seq
            user_pos[uid] = pos
            user_neg[uid] = neg
            unique_users.add(uid)
        print("all result length = {}, unique user = {}".format(len(all_results), len(unique_users)))
        print("=" * 80)
        return user_seq, user_pos, user_neg
        
        
    def get_data_of_one_user(self, uid):
        seq, pos, neg = [], [], []
        train_seq = self.user_train[uid]
        valid_item = self.user_valid[uid]
        test_item = self.user_test[uid]
        
        if self.mode == 'valid':
            seq = train_seq
            pos.extend(valid_item)
        elif self.mode == 'test':
            seq = train_seq + valid_item
            pos.extend(test_item)
        
        for _ in range(self.neg_num):
            t = np.random.randint(1, self.item_num + 1)
            while t in set(seq) | set(pos) | set(neg):
                t = np.random.randint(1, self.item_num + 1)
            neg.append(t)
        return (uid, seq, pos, neg)
                

    def __len__(self):
        return len(self.user_lst)
    
    
    def __getitem__(self, index):
        uid = self.user_lst[index]
        seq = self.user_seq[uid]
        pos = self.user_pos[uid]
        neg = self.user_neg[uid]

        if self.padding_mode.lower() == 'left':
            seq = [0] * (self.max_len - len(seq)) + seq
            seq = seq[-self.max_len:]
            return torch.from_numpy(np.array(uid)), torch.from_numpy(np.array(seq)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg))
        elif self.padding_mode.lower() == 'right':
            seq_len = [min(self.max_len, len(seq))]
            seq = seq + [0] * (self.max_len - len(seq))
            seq = seq[-self.max_len:]
            return torch.from_numpy(np.array(uid)), torch.from_numpy(np.array(seq)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg)), torch.from_numpy(np.array(seq_len))
        



# SASRec Dataset Class
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()