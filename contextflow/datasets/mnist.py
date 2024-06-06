import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as v2F


class MNISTR(datasets.MNIST):
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

        self.context  = context[0]
        self.contexts = contexts[0]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, context) where target/context is index of the target/context class.
        """
        img, target = self.data[index], self.targets[index]
        #print(img.shape, img.dtype)
        img = img.unsqueeze(0)
        
        if self.transform is not None:
            img = self.transform(img)

        #print(img.shape, img.dtype, torch.min(img), torch.max(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.context == -1:  # generalist doesn't have a context
            context = torch.zeros(1, dtype=torch.long, device=img.device)
        elif self.context == self.contexts:  # context for all specialists
            context = torch.randint(self.contexts, (1,), dtype=torch.long, device=img.device)
            img = v2F.rotate(img, 360.0*context/self.contexts)
        else:  # self.context >= 0 and self.context < self.contexts:  # context for a single specialist
            context = torch.tensor([self.context], dtype=torch.long, device=img.device)
            img = v2F.rotate(img, 360.0*context/self.contexts)
        
        return img, target, context


def load_data(c, eval_context=0, contexts=0):
    #assert c.data_aug == True
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    train_transform = torch.nn.Sequential(v2.Pad(2)).to(c.device)

    test_transform = train_transform  # transforms.Compose([ToTensorNoNorm()])
    train_context = contexts  # test always with contexts
    test_context = eval_context  # test always with contexts
    print('test_context/train_context:', test_context, train_context)
    data_train = MNISTR('../data', train=True,  transform=train_transform, download=True, context=train_context, contexts=contexts)
    data_val   = MNISTR('../data', train=True,  transform=test_transform,  download=True, context=train_context, contexts=contexts)
    data_test  = MNISTR('../data', train=False, transform=test_transform,  download=True, context=test_context,  contexts=contexts)

    trainset = Subset(data_train, torch.arange(0, 50000))
    valset   = Subset(data_val,   torch.arange(50000, 60000))
    testset  = data_test
 
    train_loader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True,  drop_last= True, **kwargs)
    val_loader   = DataLoader(valset,   batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    test_loader  = DataLoader(testset,  batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    print('train/val/test loader length',  len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print('train/val/test loader batches', len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader
