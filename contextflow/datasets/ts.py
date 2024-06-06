import os, random, time, math
import pandas as pd
import numpy as np
from typing import cast
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.utils import shuffle
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold)
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + sigma*torch.randn_like(x)

def scaling(x, sigma=0.1):  # BxCxT
    # https://arxiv.org/pdf/1706.00527.pdf
    s = 1.0 + sigma*torch.rand((x.shape[0], x.shape[1], 1), device=x.device)
    return s*x

'''def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = torch.arange(x.shape[2])
    num_segs = torch.randint(1, max_segments, (x.shape[0],))
    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = 1+torch.randperm(x.shape[2]-2)[:num_segs[i]-1]
                splits = torch.tensor_split(orig_steps, split_points.sort()[0].tolist())
            else:
                splits = torch.tensor_split(orig_steps, num_segs[i])
            warp = torch.concatenate(tuple(random.sample(splits, len(splits))))
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return ret'''

class InputDropout(nn.Module):
    """
    Removes input vectors with a probability of inp_dp_rate
    """
    def __init__(self, dp_rate, scale_features=False):
        super().__init__()
        self.dp_rate = dp_rate
        self.scale_features = scale_features

    def forward(self, x):
        if not self.training: return x
        else:
            dp_mask = x.new_zeros(x.size(0), x.size(1), 1)
            #dp_mask = torch.zeros_like(x)
            dp_mask.bernoulli_(p=self.dp_rate)
            x = x * (1 - dp_mask)
            if self.scale_features: x = x * 1.0 / (1.0 - self.dp_rate)            
            return x


class ATMdataset(Dataset):
    def __init__(self, c):
        self.L = c.window_length
        self.supervision = c.supervision
        data_path = os.path.join('../data', c.dataset)
        if not os.path.exists(data_path): raise Exception('{} data path for {} is not found'.format(data_path, c.dataset))
        data = np.load(os.path.join(data_path, 'sigma_pdm.npy'), allow_pickle=True)
        assert data.shape[-1] == self.L, 'time series length is incorrect'
        print('ATM data shape:', data.shape)
        # all data
        X = data[:, :-4, :]
        y = data[:, -1, -1]
        g = data[:, -3, -1]
        gDict = {e:i for i,e in enumerate(set(g))}
        g = np.array([gDict[e] for e in g])  # remap to context
        # split
        split_file  = os.path.join(data_path, 'split_{}_{}_{}.npz'.format(c.supervision, c.fold, c.seed))
        if os.path.isfile(split_file):
            split = np.load(split_file, allow_pickle=False)
            train_idx, valid_idx = split['train_idx'], split['test_idx']
            print('Reading split indices from {} (sum={}/{})'.format(split_file, np.sum(train_idx), np.sum(valid_idx)))
            self.valid_idx = list(valid_idx)
            self.train_idx = list(train_idx)
        else:
            sgk = StratifiedKFold(n_splits=c.folds, shuffle=True, random_state=c.seed)
            self.train_idx, self.valid_idx = list(sgk.split(X, y=y))[c.fold]
            # supervision settings:
            if self.supervision == 'weak':
                pos = list(self.train_idx[y[self.train_idx]==0])
                self.train_idx = pos  # remove all anomalies from the train
            elif self.supervision == 'subs':
                neg = list(self.train_idx[y[self.train_idx]==1])
                pos = list(self.train_idx[y[self.train_idx]==0])
                self.train_idx = pos + random.sample(neg, len(neg)//10)  # remove 90% of anomalies from the train
            
            print('Writing split indices from {}'.format(split_file))
            np.savez(split_file, train_idx=np.array(self.train_idx), test_idx=np.array(self.valid_idx))
        # preprocessor:
        if   c.preprocessing == "minmax":   est = MinMaxScaler()
        elif c.preprocessing == "standard": est = StandardScaler()
        elif c.preprocessing == "robust":   est = RobustScaler()
        X = est.fit_transform(rearrange(X, 'b d l -> (b l) d'))
        X = rearrange(X, '(b l) d -> b d l', l=self.L)
        # pytorch stuff:
        self.X       = torch.tensor(X, dtype=torch.float).unsqueeze(-1)  # NxDxTx1
        self.target  = torch.tensor(y, dtype=torch.long)  # binary label  # N
        self.context = torch.tensor(g, dtype=torch.long).unsqueeze(-1)  # machine ID Nx1
        self.inpdp = InputDropout(0.1, scale_features=True)
        #print('train/test contexts:', len(set(g[self.train_idx])), len(set(g[self.valid_idx])))
        assert len(set(g[self.train_idx]) - set(g[self.valid_idx])) == 0, 'train/eval sets should have all contexts: {}'.format(set(g[self.train_idx]) - set(g[self.valid_idx]))

    def __getitem__(self, index):
        x = self.X[index]
        #if index in self.train_idx:
        #    x = self.inpdp(self.X[index])
        return x, self.target[index], self.context[index]

    def __len__(self):
        return len(self.target)

from datasets.mtad_data_preprocess import preprocessor, generate_windows
from datasets.mtad_dataloader import data_path_dict, mtad_entities, load_dataset, sliding_window_dataset

def load_data_ts(c, context=0, contexts=0):
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    # data
    if c.dataset == 'atm':
        full_dataset  = ATMdataset(c)
        train_dataset = Subset(full_dataset, full_dataset.train_idx)
        valid_dataset = Subset(full_dataset, full_dataset.valid_idx)
        train_weight = torch.tensor([1., math.log(len(train_dataset) / sum(full_dataset.target[full_dataset.train_idx]))], device=c.device)
        valid_weight = torch.tensor([1., math.log(len(valid_dataset) / sum(full_dataset.target[full_dataset.valid_idx]))], device=c.device)
    elif c.dataset in ['msl', 'smd', 'smap']:
        train_context = context  # train with certain contexts
        test_context  = context  # test with certain contexts
        entities = mtad_entities[c.dataset]
        assert len(entities) == contexts[-1], 'Number of contexts should be equal to number of entities'
        data_dict = load_dataset(data_root = data_path_dict[c.dataset], entities = entities, dim = c.data_vars[0], valid_ratio = 0,
                                    test_label_postfix = "test_label.pkl", test_postfix = "test.pkl", train_postfix = "train.pkl")
        # preprocessing
        pp = preprocessor()
        data_dict = pp.normalize(data_dict, method=c.preprocessing)
        # sliding windows
        window_dict = generate_windows(data_dict, window_size=c.window_length, stride=1)
        # batch data
        train_dataset = sliding_window_dataset(c, window_dict, train=True,  context=train_context, contexts=contexts)
        valid_dataset = sliding_window_dataset(c, window_dict, train=False, context=test_context , contexts=contexts)
        train_weight = torch.ones(1, device=c.device)
        valid_weight = torch.ones(1, device=c.device)
        assert len(set(train_dataset.context.flatten().tolist()) - set(valid_dataset.context.flatten().tolist())) == 0, 'train/eval sets should have all contexts: {}'.format(set(train_dataset.context.flatten().tolist()) - set(valid_dataset.context.flatten().tolist()))
    else: raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    #
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,  drop_last=True,  prefetch_factor=8, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, prefetch_factor=8, **kwargs)
    if c.verbose: print('train/val loader length',  len(train_loader.dataset), len(valid_loader.dataset))
    if c.verbose: print('train/val loader batches', len(train_loader), len(valid_loader))
    if c.verbose: print('train/val loader weights', train_weight, valid_weight)

    return (train_loader, valid_loader), (train_weight, valid_weight)