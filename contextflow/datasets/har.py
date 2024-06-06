import os, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + sigma*torch.randn_like(x)

def scaling(x, sigma=0.1):  # BxCxT
    # https://arxiv.org/pdf/1706.00527.pdf
    s = 1.0 + sigma*torch.rand((x.shape[0], x.shape[1], 1), device=x.device)
    return s*x

'''def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret'''

def permutation(x, max_segments=5, seg_mode="equal"):
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
    return ret


def DataTransform(sample, c):
    weak_aug = scaling(sample.clone(), c.jitter_scale_ratio)
    #ts_sample = sample.clone().permute(0, 2, 1).numpy()  # BxCxT -> BxTxC
    #p_sample = permutation(ts_sample, max_segments=c.max_seg)
    #j_sample = torch.from_numpy(p_sample).permute(0, 2, 1)  # BxTxC -> BxCxT
    #strong_aug = jitter(j_sample, c.jitter_ratio)
    strong_aug = jitter(permutation(sample.clone(), max_segments=c.max_seg), c.jitter_ratio)
    return weak_aug, strong_aug


class HARdataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, c, data, percentage=100, aug=False):
        super(HARdataset, self).__init__()
        self.L = c.ts_max_length
        self.C = c.cont_features
        self.D = c.disc_features
        self.supervision = c.supervision
        self.aug = aug

        X = data["samples"]
        y = data["labels"]
        g = data["context"]

        assert X.shape[ 1] == self.C, 'number continuous features is incorrect'
        assert X.shape[-1] == self.L, 'time series length is incorrect'

        # split
        #split_file  = os.path.join(c.data_path, 'split_{}_{}_{}.npz'.format(c.supervision, c.fold, c.seed))
        #self.is_npz = os.path.isfile(split_file)
        #if self.is_npz:
        #    split = np.load(split_file, allow_pickle=False) if self.is_npz else None
        #    train_idx, valid_idx = split['train_idx'], split['test_idx']
        #    print('Reading split indices from {} (sum={}/{})'.format(split_file, np.sum(train_idx), np.sum(valid_idx)))
        #    self.valid_idx = list(valid_idx)
        #    self.train_idx = list(train_idx)
        #else:
        if percentage < 100:
            num_samples = len(y)*(100-percentage)//100
            idx = torch.multinomial(torch.ones(num_samples)/len(y), num_samples=num_samples)
            y[idx] = -1
            print('percentage:', percentage, torch.sum(y!=-1), num_samples, len(y))

        x_data = X.float()
        if aug:  # no need to apply Augmentations in other modes
            x_aug1, x_aug2 = DataTransform(x_data, c)
            self.x_aug1 = x_aug1.unsqueeze(-1)
            self.x_aug2 = x_aug2.unsqueeze(-1)

        self.x_data = x_data.unsqueeze(-1)  # BxCxTx1
        self.y_data = y.long()
        self.g_data = g.long().unsqueeze(-1)

        self.translation = 0.5
        self.scale       = 2*torch.tensor((np.max(np.abs(self.x_data.numpy()), axis=(0,2,3))).tolist(), dtype=torch.float)
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        if self.aug:
            return (self.x_data[index], self.x_aug1[index], self.x_aug2[index]), self.y_data[index], self.g_data[index]
        else:
            return  self.x_data[index], self.y_data[index], self.g_data[index]

    def __len__(self):
        return self.len


def load_data_har(c, C=0, D=0, L=0, percentage=100):
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    c.data_path = os.path.join('../data', c.dataset)
    #print(os.path.exists(c.data_path), c.data_path)
    if not os.path.exists(c.data_path): os.makedirs(c.data_path)
        
    c.cont_features = C
    c.disc_features = D
    c.ts_max_length = L

    # aug:
    c.jitter_scale_ratio = 1.1
    c.jitter_ratio = 0.8
    c.max_seg = 8

    # data
    train_data_file = os.path.join(c.data_path, 'train_F{}_S{}.pt'.format(c.fold, c.seed))
    valid_data_file = os.path.join(c.data_path, 'valid_F{}_S{}.pt'.format(c.fold, c.seed))
    if not (os.path.isfile(train_data_file) and os.path.isfile(valid_data_file)): preprocess_har(c, C, D, L)
    train_data = torch.load(train_data_file)
    valid_data = torch.load(valid_data_file)
    test_data  = torch.load(os.path.join(c.data_path, 'test.pt'))
    
    #if percentage < 100: # split to train/val/test like in https://arxiv.org/pdf/2106.14112.pdf
    train_dataset = HARdataset(c, train_data, percentage=percentage, aug=True)
    valid_dataset = HARdataset(c, valid_data)
    test_dataset  = HARdataset(c, test_data)

    #else: # use all train like in https://arxiv.org/pdf/2312.02185.pdf
    #    alltrain_data = dict()
    #    alltrain_data["samples"] = torch.concat((train_data["samples"], valid_data["samples"]))
    #    alltrain_data["labels"]  = torch.concat((train_data["labels"],  valid_data["labels"]))
    #    alltrain_data["context"] = torch.concat((train_data["context"], valid_data["context"]))
    #    train_dataset = HARdataset(c, alltrain_data)
    #    valid_dataset = test_dataset
    #    test_dataset  = HARdataset(c, test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,  drop_last= True, prefetch_factor=8, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, prefetch_factor=8, **kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=c.batch_size, shuffle=False, drop_last=False, prefetch_factor=8, **kwargs)
    print('train/val/test loader length',  len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset))
    print('train/val/test loader batches', len(train_loader), len(valid_loader), len(test_loader))

    return (train_loader, valid_loader, test_loader)


def preprocess_har(c, C=0, D=0, L=0):
    data_dir   = os.path.join('../data', 'UCI HAR Dataset')
    output_dir = os.path.join(c.data_path)
    # Samples
    train_subject = np.loadtxt(f'{data_dir}/train/subject_train.txt') - 1
    train_acc_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_x_train.txt')
    train_acc_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_y_train.txt')
    train_acc_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_z_train.txt')
    train_gyro_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_x_train.txt')
    train_gyro_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_y_train.txt')
    train_gyro_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_z_train.txt')
    train_tot_acc_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_x_train.txt')
    train_tot_acc_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_y_train.txt')
    train_tot_acc_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_z_train.txt')

    test_subject = np.loadtxt(f'{data_dir}/test/subject_test.txt') - 1
    test_acc_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_x_test.txt')
    test_acc_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_y_test.txt')
    test_acc_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_z_test.txt')
    test_gyro_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_x_test.txt')
    test_gyro_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_y_test.txt')
    test_gyro_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_z_test.txt')
    test_tot_acc_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_x_test.txt')
    test_tot_acc_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_y_test.txt')
    test_tot_acc_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_z_test.txt')

    # Stacking channels together data
    train_data = np.stack((train_acc_x, train_acc_y, train_acc_z,
                        train_gyro_x, train_gyro_y, train_gyro_z,
                        train_tot_acc_x, train_tot_acc_y, train_tot_acc_z), axis=1)
    test_data = np.stack((test_acc_x, test_acc_y, test_acc_z,
                        test_gyro_x, test_gyro_y, test_gyro_z,
                        test_tot_acc_x, test_tot_acc_y, test_tot_acc_z), axis=1)
    # labels
    train_labels = np.loadtxt(f'{data_dir}/train/y_train.txt') - 1
    test_labels  = np.loadtxt(f'{data_dir}/test/y_test.txt') - 1

    #data    = np.concatenate((train_data   , test_data   ), axis=0)
    #labels  = np.concatenate((train_labels , test_labels ), axis=0)
    #subject = np.concatenate((train_subject, test_subject), axis=0)

    data, labels, subject = train_data, train_labels, train_subject
    sgk = StratifiedGroupKFold(n_splits=c.folds, shuffle=True, random_state=c.seed)
    for fold in range(c.folds):
        #X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(train_data, train_labels, train_subject, test_size=0.2, random_state=42)
        train_idx, val_idx = list(sgk.split(data, y=labels, groups=subject))[fold]

        dat_dict = dict()
        dat_dict["samples"] = torch.from_numpy(train_data[train_idx])
        dat_dict["labels"]  = torch.from_numpy(train_labels[train_idx])
        dat_dict["context"] = torch.from_numpy(train_subject[train_idx])
        torch.save(dat_dict, os.path.join(output_dir, 'train_F{}_S{}.pt'.format(fold, c.seed)))

        dat_dict = dict()
        dat_dict["samples"] = torch.from_numpy(train_data[val_idx])
        dat_dict["labels"]  = torch.from_numpy(train_labels[val_idx])
        dat_dict["context"] = torch.from_numpy(train_subject[val_idx])
        torch.save(dat_dict, os.path.join(output_dir, 'valid_F{}_S{}.pt'.format(fold, c.seed)))

    dat_dict = dict()
    dat_dict["context"] = torch.from_numpy(test_subject)
    dat_dict["samples"] = torch.from_numpy(test_data)
    dat_dict["labels"] = torch.from_numpy(test_labels)
    torch.save(dat_dict, os.path.join(output_dir, 'test.pt'))