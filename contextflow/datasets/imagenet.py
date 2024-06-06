import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import tarfile
import os
import numpy as np
from datasets import corrupt

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class ImageFolderC(datasets.DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        context =-1,
        contexts=0,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, context) where target/context is class_index of the target/context class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.context == self.contexts:  # context for all specialists
            corruption = int(torch.randint(self.contexts[0], (1,)))
            severity   = int(torch.randint(self.contexts[1], (1,))) + 1
            context = torch.tensor([corruption, severity], dtype=torch.long)
        elif self.context == -1:  # generalist doesn't have a context
            corruption = int(torch.randint(self.contexts[0], (1,)))
            severity   = int(torch.randint(self.contexts[1], (1,))) + 1
            context = torch.tensor([0, 0], dtype=torch.long)
        else:  # context for a single specialist
            corruption = int(self.context[0])
            severity   = int(self.context[1])
            context = torch.tensor([corruption, severity], dtype=torch.long)
        print(sample.shape, corruption, severity)
        sample = corrupt(sample, corruption_number=corruption, severity=severity)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, context


def extract_tar(tarpath):
    assert tarpath.endswith('.tar')
    startdir = tarpath[:-4] + '/'
    if os.path.exists(startdir): return startdir
    print('Extracting', tarpath)
    with tarfile.open(name=tarpath) as tar:
        t = 0
        done = False
        while not done:
            path = os.path.join(startdir, 'images{}'.format(t))
            os.makedirs(path, exist_ok=True)
            print(path)
            for i in range(50000):
                member = tar.next()
                if member is None:
                    done = True
                    break
                # Skip directories
                while member.isdir():
                    member = tar.next()
                    if member is None:
                        done = True
                        break

                member.name = member.name.split('/')[-1]
                tar.extract(member, path=path)
            t += 1
    return startdir


def load_data(c, context=0, contexts=0, resolution=32, data_dir='~/data/imagenet_32x32'):
    assert resolution == 32 or resolution == 64
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    trainpath = f'{data_dir}/train_{resolution}x{resolution}.tar'
    valpath = f'{data_dir}/valid_{resolution}x{resolution}.tar'

    trainpath = extract_tar(trainpath)
    valpath   = extract_tar(valpath)

    data_transform = transforms.Compose([ToTensorNoNorm()])

    print('Starting loading ImageNet')

    data_train = ImageFolderC(trainpath, transform=data_transform, context=context, contexts=contexts)
    #data_val   = ImageFolderC(trainpath, transform=data_transform, context=context, contexts=contexts)
    data_test  = ImageFolderC(valpath,   transform=data_transform, context=context, contexts=contexts)

    print('Number of trainval/test images', len(data_train), len(data_test))

    val_idcs   = np.random.choice(len(data_train), size=20000, replace=False)
    train_idcs = np.setdiff1d(np.arange(len(data_train)), val_idcs)

    trainset = Subset(data_train, train_idcs)
    valset   = Subset(data_train, val_idcs)  # data_val
    testset  = data_test
    
    train_loader = DataLoader(trainset, batch_size=c.batch_size, shuffle=True,  drop_last=True,  **kwargs)
    val_loader   = DataLoader(valset,   batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    test_loader  = DataLoader(testset,  batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)

    return train_loader, val_loader, test_loader
