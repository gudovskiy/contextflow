from __future__ import print_function
import argparse

__all__ = ['get_args']

def get_args():
    parser = argparse.ArgumentParser(description='ContextFlow++')
    parser.add_argument('--dataset', default='mnist', type=str, choices={'mnist', 'cifar10', 'atm', 'msl', 'smd', 'smap'}, help='dataset name (default: mnist)')
    parser.add_argument('--supervision', default='full', type=str, metavar='D', help='full/weak/subs (default: full)')
    parser.add_argument('--batch-size', '-bs', default=256, type=int, metavar='B', help='train batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=48, metavar='N', help='number of epochs to train (default: 36)')
    parser.add_argument('--workers', default=4, type=int, metavar='G', help='number of data loading workers (default: 4)')
    parser.add_argument('--gpu', default='0', type=str, metavar='G', help='GPU device number')
    parser.add_argument('--coupling', default='eye', type=str, choices={'eye', 'conv', 'trans', 'maf'}, help='coupling blocks: conv/trans/eye (default: conv)')
    parser.add_argument('--dist', default='gauss', type=str, choices={'gauss', 'tdist'}, help='base distribution: gauss/tdist (default: gauss)')
    parser.add_argument('--action-type', default='train-generalist', type=str, choices={'train-specialist', 'train-generalist', 'test-specialist', 'test-generalist'}, help='mode (default: train-generalist)')
    parser.add_argument('--enc-type', default='uniform', type=str, choices={'probsample', 'eyesample', 'uniform', 'vardeq', 'argmax'}, help='mode (default: uniform)')
    parser.add_argument('--enc-emb', default='onehot', type=str, choices={'eye', 'onehot', 'embed'}, help='mode (default: onehot)')
    parser.add_argument('--contextflow', action='store_true', default=False, help='enables ContextFlow++')
    parser.add_argument('--clean', action='store_true', default=False, help='enables clean (undistorted model)')
    parser.add_argument('--wandb', action='store_true', default=False, help='enables WanDB')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', default=2, type=int, metavar='D', help='seed for dataset split (default: 2)')
    parser.add_argument('--fold', default=0, type=int, metavar='D', help='fold index for dataset split (default: 0)')
    parser.add_argument('--folds', default=5, type=int, metavar='D', help='number of folds for dataset split (default: 5)')
    parser.add_argument('--checkpoint', default=None, type=str, help='resume from the checkpoint')
    parser.add_argument('--save-checkpoint', action='store_true', default=True, help='saves best checkpoints')
    parser.add_argument('--verbose', action='store_true', default=False, help='enables logging')
    parser.add_argument('--complexity', action='store_true', default=False, help='prints complexity using torchinfo')
    args = parser.parse_args()
    return args
