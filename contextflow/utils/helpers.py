import torch
import os, random, time, math
import numpy as np

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_child_parameters(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        freeze_child_parameters(child)


def freeze_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


# From http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations)
        """
        if data is not None:
            data = np.array(data)
            print(data.shape)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.array(data)

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n