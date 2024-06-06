import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from config import get_args
c = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
c.use_cuda = not c.no_cuda and torch.cuda.is_available()
c.device = torch.device("cuda" if c.use_cuda else "cpu")
kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
c.dataset = 'har'
c.cont_features = 9
c.disc_features = 0
c.ts_max_length = 128
from datasets.har import HARdataset
c.data_path = os.path.join('../data', c.dataset)
train_data = torch.load(os.path.join(c.data_path, "train.pt"))
valid_data = torch.load(os.path.join(c.data_path, "val.pt"))
test_data  = torch.load(os.path.join(c.data_path, "test.pt"))
train_dataset = HARdataset(c, train_data)
valid_dataset = HARdataset(c, valid_data)
test_dataset  = HARdataset(c, test_data)
train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,  drop_last= True, prefetch_factor=8, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=c.batch_size, shuffle=False, drop_last= True, prefetch_factor=8, **kwargs)
test_loader  = DataLoader(test_dataset,  batch_size=c.batch_size, shuffle=False, drop_last=False, prefetch_factor=8, **kwargs)