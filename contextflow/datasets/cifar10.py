import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as v2F
import math
from datasets import corrupt
import numpy as np


class CIFAR10C(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        context  = [0],
        contexts = [0],
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.context  = context
        self.contexts = contexts

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, context) where target/context is index of the target/context class.
        """
        img, target = self.data[index], self.targets[index]

        if any([c == -1 for c in self.context]):  # generalist doesn't have a context
            context = torch.zeros(2, dtype=torch.long)
        elif all([c1 == c2 for c1,c2 in zip(self.context, self.contexts)]):  # context for all specialists
            corruption = torch.randint(0, self.contexts[0]  , (1,), dtype=torch.long)
            severity   = torch.randint(1, self.contexts[1]+1, (1,), dtype=torch.long)
            context = torch.cat((corruption, severity-1))
            img = corrupt(img, corruption_number=int(corruption), severity=int(severity))
        else:  # context for a single specialist
            corruption = torch.tensor([self.context[0]]  , dtype=torch.long)
            severity   = torch.tensor([self.context[1]+1], dtype=torch.long)
            context = torch.cat((corruption, severity-1))
            img = corrupt(img, corruption_number=int(corruption), severity=int(severity))

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, context


def load_data(c, eval_context=0, contexts=0):
    #assert c.data_aug == True
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    train_transform = torch.nn.Sequential(v2.PILToTensor()).to(c.device)

    test_transform = train_transform  # transforms.Compose([ToTensorNoNorm()])
    train_context = contexts  # train with all contexts
    test_context = eval_context  # test with certain contexts
    print('test_context/train_context:', test_context, train_context)

    data_train = CIFAR10C('../data', train=True,  transform=train_transform, download=True, context=train_context, contexts=contexts)
    data_val   = CIFAR10C('../data', train=True,  transform=test_transform,  download=True, context=train_context, contexts=contexts)
    data_test  = CIFAR10C('../data', train=False, transform=test_transform,  download=True, context=test_context,  contexts=contexts)

    trainset = Subset(data_train, torch.arange(0, 40000))
    valset   = Subset(data_val,   torch.arange(40000, 50000))
    testset  = data_test
 
    train_loader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True,  drop_last= True, **kwargs)
    val_loader   = DataLoader(valset,   batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    test_loader  = DataLoader(testset,  batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    print('train/val/test loader length',  len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print('train/val/test loader batches', len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader
