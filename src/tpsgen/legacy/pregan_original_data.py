from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
from torch import squeeze
import pandas as pd
import torch

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class LoadData(Dataset):

    def __init__(self, path='data/ecoli_100_space_fix.csv', split_r=0.9, is_train=True, gpu_ids='0'):
        data = pd.read_csv(path)  # 加载整个数据文件
        realB = list(data['realB'])
        realA = list(data['realA'])
        expr = list(data['expr'])  # 加载 expr 列

        data_size = len(realB)
        split_idx = int(data_size * split_r)
        self.gpu_ids = gpu_ids

        if is_train:
            st, ed = 0, split_idx
        else:
            st, ed = split_idx, data_size

        self.storage, self.input_seq, self.expr = [], [], []
        for i in range(st, ed, 1):
            self.storage.append(one_hot(realB[i].split('\n')[0].upper()))
            self.input_seq.append(backbone_one_hot(realA[i].split('\n')[0].upper()))
            self.expr.append(expr[i])  # 保存 expr 值

    def __getitem__(self, item):
        in_seq = torch.from_numpy(self.input_seq[item]).unsqueeze(0)
        label_seq = torch.from_numpy(self.storage[item]).unsqueeze(0)
        expr_value = self.expr[item]  # 获取 expr 值

        if len(self.gpu_ids) > 0:
            return {
                'in': in_seq[0, :].float().cuda(),
                'out': squeeze(label_seq).float().cuda(),
                'expr': torch.tensor(expr_value, dtype=torch.float).cuda()  # 将 expr 转为 Tensor
            }
        else:
            return {
                'in': in_seq[0, :].float(),
                'out': squeeze(label_seq).float(),
                'expr': torch.tensor(expr_value, dtype=torch.float)  # 将 expr 转为 Tensor
            }

    def __len__(self):
        return len(self.storage)


def one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        encoded[charmap[seq[i]], i] = 1
    return encoded


def backbone_one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            encoded[charmap[seq[i]], i] = 1
    return encoded
