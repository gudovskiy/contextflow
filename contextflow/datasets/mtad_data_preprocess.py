import logging
import os
import pickle
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class preprocessor:
    def __init__(self):
        self.vocab_size = None
        self.discretizer_list = defaultdict(list)

    def save(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        logging.info("Saving preprocessor into {}".format(filepath))
        with open(filepath, "wb") as fw:
            pickle.dump(self.__dict__, fw)

    def load(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        logging.info("Loading preprocessor from {}".format(filepath))
        with open(filepath, "rb") as fw:
            self.__dict__.update(pickle.load(fw))

    def normalize(self, data_dict, method="minmax"):
        if method == "none":
            return data_dict
        logging.info("Normalizing data with {}".format(method))
        normalized_dict = defaultdict(dict)
        for k, subdata_dict in data_dict.items():
            # method: minmax, standard, robust
            # fit_transform using train
            if method == "minmax":
                est = MinMaxScaler()
            elif method == "standard":
                est = StandardScaler()
            elif method == "robust":
                est = RobustScaler()

            train_ = est.fit_transform(subdata_dict["train"])
            test_ = est.transform(subdata_dict["test"])

            # assign back
            normalized_dict[k]["train"] = train_
            normalized_dict[k]["test"] = test_
            for subk in subdata_dict.keys():
                if subk not in ["train", "test"]:
                    normalized_dict[k][subk] = subdata_dict[subk]
        return normalized_dict

# not circular. hence, clipping happens
def get_windows(ts, labels=None, window_size=128, stride=1, dim=None):
    i = 0
    ts_len = ts.shape[0]
    windows = []
    label_windows = []
    while i < ts_len:
        if i < window_size -1:  # replication padding as in TranAD
            windows.append(np.concatenate((np.repeat(ts[0].reshape(1,-1), window_size-i-1, axis=0), ts[:i+1])))
        else:
            windows.append(ts[i-window_size+1:i+1])
        
        if labels is not None: label_windows.append(labels[i])
        i += stride
    if labels is not None:
        return np.array(windows, dtype=float), np.array(label_windows, dtype=int)
    else:
        return np.array(windows, dtype=float), None


def generate_windows(data_dict, window_size=100, nrows=None, stride=1, **kwargs):
    logging.info("Generating sliding windows (size {}).".format(window_size))
    results = defaultdict(dict)
    for dataname, subdata_dict in data_dict.items():
        for k in ["train", "valid", "test"]:
            if k not in subdata_dict: continue
            data = subdata_dict[k][0:nrows]
            #print('data:', data.shape)
            if k == "train":
                data_windows, _ = get_windows(data, window_size=window_size, stride=stride)
                results[dataname]["train_windows"] = data_windows
            if k == "valid":
                data_windows, _ = get_windows(data, window_size=window_size, stride=stride)
                results[dataname]["valid_windows"] = data_windows
            if k == "test":
                test_label = subdata_dict["test_label"][0:nrows]
                test_windows, test_label = get_windows(data, test_label, window_size=window_size, stride=1)
                results[dataname]["test_windows"] = test_windows
                results[dataname]["test_label"] = test_label
            logging.info("Windows for {} #: {}".format(k, data_windows.shape))

    return results
