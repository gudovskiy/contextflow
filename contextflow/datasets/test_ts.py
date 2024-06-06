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
c.dataset = 'msl'
c.contexts = [27]
c.cont_features = 55
c.disc_features = 0
c.window_length = 10
train_context = c.contexts  # train with all contexts
test_context = c.contexts  # test with certain contexts
from datasets.mtad_data_preprocess import preprocessor, generate_windows
from datasets.mtad_dataloader import data_path_dict, mtad_entities, load_dataset, sliding_window_dataset
entities = mtad_entities[c.dataset]
assert len(entities) == c.contexts[-1], 'Number of contexts should be equal to number of entities'
data_dict = load_dataset(data_root = data_path_dict[c.dataset], entities = entities, dim = c.cont_features+c.disc_features, 
                         valid_ratio = 0, test_label_postfix = "test_label.pkl", test_postfix = "test.pkl", train_postfix = "train.pkl")
# preprocessing
pp = preprocessor()
data_dict = pp.normalize(data_dict, method="minmax")
# sliding windows
window_dict = generate_windows(data_dict, window_size=c.window_length, stride=1)
# batch data
train_dataset = sliding_window_dataset(c, window_dict, train=True,  context=train_context, contexts=c.contexts)
valid_dataset = sliding_window_dataset(c, window_dict, train=False, context=test_context , contexts=c.contexts)



train_dataset, valid_dataset = TranADdataset(c)
train_weight = torch.tensor([1., math.log(len(train_dataset) / sum(full_dataset.target[full_dataset.train_idx]))], device=c.device)
valid_weight = torch.tensor([1., math.log(len(valid_dataset) / sum(full_dataset.target[full_dataset.valid_idx]))], device=c.device)
train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,  drop_last= True, prefetch_factor=8, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, prefetch_factor=8, **kwargs)
print('train/val loader length',  len(train_loader.dataset), len(valid_loader.dataset))
print('train/val loader batches', len(train_loader), len(valid_loader))
print('train/val loader weights', train_weight, valid_weight)