import logging
import os
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torch
from datasets.ts import InputDropout, jitter, scaling


data_path_dict = {
    "smd": "../data/processed_SMD",
    "smap": "../data/processed_SMAP",
    "msl": "../data/processed_MSL",
    #"wadi": "./datasets/anomaly/WADI/processed",
    #"swat": "./datasets/anomaly/SWAT/processed",
    #"wadi_split": "./datasets/anomaly/WADI_SPLIT/processed",
    #"swat_split": "./datasets/anomaly/SWAT_SPLIT/processed",
}

mtad_entities = {
    "smd": ["machine-1-{}".format(i) for i in range(1,  9)] + 
           ["machine-2-{}".format(i) for i in range(1, 10)] + 
           ["machine-3-{}".format(i) for i in range(1, 12)],

    "smap": ["P-1","S-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","E-8","E-9","E-10","E-11","E-12","E-13","A-1","D-1",
             "P-2","P-3","D-2","D-3","D-4","A-2","A-3","A-4","G-1","G-2","D-5","D-6","D-7","F-1","P-4","G-3","T-1","T-2","D-8","D-9","F-2","G-4",
             "T-3","D-11","D-12","B-1","G-6","G-7","P-7","R-1","A-5","A-6","A-7","D-13","P-2","A-8","A-9","F-3",
    ],
    "msl": [
        "C-1", "C-2",
        "D-14", "D-15", "D-16",
        "F-4", "F-5", "F-7", "F-8",
        "M-1", "M-2", "M-3", "M-4", "M-5", "M-6", "M-7",
        "P-10", "P-11", "P-14", "P-15",
        "S-2",
        "T-4", "T-5", "T-8", "T-9", "T-12", "T-13",
    ],
    #"wadi": ["wadi"],
    #"swat": ["swat"],
    #"wadi_split": ["wadi-1", "wadi-2", "wadi-3"],  # if OOM occurs
}


def load_dataset(
    data_root,
    entities,
    valid_ratio,
    dim,
    test_label_postfix,
    test_postfix,
    train_postfix,
    nan_value=0,
    nrows=None,
):
    """
    use_dim: dimension used in multivariate timeseries
    """
    logging.info("Loading data from {}".format(data_root))

    data = defaultdict(dict)
    total_train_len, total_valid_len, total_test_len = 0, 0, 0
    for dataname in entities:
        with open(os.path.join(data_root, "{}_{}".format(dataname, train_postfix)), "rb") as f:
            train = pickle.load(f).reshape((-1, dim))[0:nrows, :]
            if valid_ratio > 0:
                split_idx = int(len(train) * valid_ratio)
                train, valid = train[:-split_idx], train[-split_idx:]
                data[dataname]["valid"] = np.nan_to_num(valid, nan_value)
                total_valid_len += len(valid)
            data[dataname]["train"] = np.nan_to_num(train, nan_value)
            total_train_len += len(train)
            #print(dataname, total_train_len)
        with open(os.path.join(data_root, "{}_{}".format(dataname, test_postfix)), "rb") as f:
            test = pickle.load(f).reshape((-1, dim))[0:nrows, :]
            data[dataname]["test"] = np.nan_to_num(test, nan_value)
            total_test_len += len(test)
            #print(dataname, total_test_len)
        with open(os.path.join(data_root, "{}_{}".format(dataname, test_label_postfix)), "rb") as f:
            data[dataname]["test_label"] = pickle.load(f).reshape(-1)[0:nrows]
    logging.info("Loading {} entities done.".format(len(entities)))
    logging.info(
        "Train/Valid/Test: {}/{}/{} lines.".format(
            total_train_len, total_valid_len, total_test_len
        )
    )

    return data


class sliding_window_dataset(Dataset):
    def __init__(self, c, data_dict, train, context, contexts):
        X, y, g = [], [], []
        if all([c1 == c2 for c1,c2 in zip(context, contexts)]):  # context for all specialists
              contexts = [i for i in range(contexts[0])]
        else: contexts = context
        if c.verbose: print('contexts:', 'train' if train else 'valid', [mtad_entities[c.dataset][i] for i in contexts])
        total_len = 0
        for cur_context in contexts:
            entity = mtad_entities[c.dataset][cur_context]
            data = data_dict[entity]["train_windows"] if train else data_dict[entity]["test_windows"]
            total_len += len(data)
            X.extend(data)
            y.extend(np.zeros(data.shape[:1], dtype=int) if train else data_dict[entity]["test_label"])
            g.extend(np.ones( data.shape[:1], dtype=int)*cur_context)
        X = np.transpose(np.array(X), axes=(0,2,1))
        y = np.array(y)
        g = np.array(g)
        #
        self.X       = torch.tensor(X, dtype=torch.float).unsqueeze(-1)  # NxDxTx1
        self.target  = torch.tensor(y, dtype=torch.long)  # N
        self.context = torch.tensor(g, dtype=torch.long).unsqueeze(-1)  # Nx1
        self.train = train
        self.inpdp = InputDropout(0.1, scale_features=True)

    def __getitem__(self, index):
        x = self.X[index]
        if self.train:
            #x = jitter(self.inpdp(self.X[index]), sigma=1e-2)
            x = self.inpdp(self.X[index])
        return x, self.target[index], self.context[index]

    def __len__(self):
        return len(self.target)


'''def get_dataloaders(train_data, test_data, valid_data=None, next_steps=0, batch_size=32, shuffle=True, num_workers=1,
):

    train_loader = DataLoader(
        sliding_window_dataset(train_data, next_steps),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        sliding_window_dataset(test_data, next_steps),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if valid_data is not None:
        valid_loader = DataLoader(
            sliding_window_dataset(valid_data, next_steps),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        valid_loader = None
    return train_loader, valid_loader, test_loader'''
