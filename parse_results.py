import os, time, copy
import numpy as np
import torch
from contextflow.config import get_args


def main(c):
    c = get_args()
    folds = c.folds
    generalist = True if 'generalist' in c.action_type else False
    if c.dataset == 'mnist':
        mixtures = 10
        num_blocks = 2
        block_size = 2
        actnorm = True
        split_prior = False
        contexts = [-1] if c.clean else [64]
    elif c.dataset == 'cifar10':
        mixtures = 10
        num_blocks = 3
        actnorm = True
        block_size = 4
        split_prior = True
        contexts = [-1, -1] if c.clean else [15, 5]
    elif c.dataset == 'atm':
        c.preprocessing = 'minmax'
        num_blocks = 3
        block_size = 2*2
        actnorm = True  # False
        split_prior = False
        contexts = [68]
        data_vars = (26, 6+6)  # (continuous, discrete+zero)
        window_length = 144
        mixtures = 2
    elif c.dataset in ['msl', 'smd', 'smap']:
        c.supervision = 'weak'
        c.preprocessing = 'minmax'
        window_length = 16
        if c.dataset == 'msl':
            num_blocks = 2
            block_size = 4
            actnorm = True
            split_prior = True
            contexts = [27]  # discrete contexts
            data_vars = (55, 0)
            mixtures = 1  # contexts[0]
        elif c.dataset == 'smd':
            num_blocks = 3
            block_size = 2*2
            actnorm = False
            split_prior = False
            contexts = [28]  # discrete contexts
            data_vars = (38, 0)
            mixtures = 1  # contexts[0]
        elif c.dataset == 'smap':
            num_blocks = 3
            block_size = 2*2
            actnorm = False
            split_prior = False
            contexts = [55]  # discrete contexts
            data_vars = (25, 0)
            mixtures = 1  # contexts[0]
    else: raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    # upd config:
    if generalist:
        context = [-1] if c.clean else contexts
        postfix = 'generalist_clean' if c.clean else 'generalist'
        #mixtures = 1 if ad_dataset
    else:
        context = contexts
        context_type = 'contextflow' if c.contextflow else 'conventional'
        postfix = 'specialist_{}_{}_{}_{}'.format(contexts, c.enc_emb, c.enc_type, context_type)
    
    if c.dataset == 'atm':
        metrics = 6
    elif c.dataset in ['msl', 'smd', 'smap']:
        metrics = 4
    else: metrics = 1
    
    results = np.zeros((metrics, folds))
    
    for fold in range(folds):
        run_name = '{}_{}_{}L_{}B_{}_{}_F{}'.format(c.dataset, c.supervision, num_blocks, block_size, c.coupling, c.dist, fold)
        checkpoint_path = os.path.join('./checkpoints', '{}_{}.pt'.format(run_name, postfix))
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        summary = checkpoint['summary']
        if c.dataset == 'atm':
            results[0][fold] = summary['Best Val ACC']
            results[1][fold] = summary['Best Val BACC']
            results[2][fold] = summary['Best Val AUC']
            results[3][fold] = summary['Best Val AP']
            results[4][fold] = summary['Best Val F1']
            results[5][fold] = summary['Best Val MS']
        elif c.dataset in ['msl', 'smd', 'smap']:
            results[0][fold] = summary['Best Val P']
            results[1][fold] = summary['Best Val R']
            results[2][fold] = summary['Best Val AUC']
            results[3][fold] = summary['Best Val F1']    
        else:
            results[0][fold] = summary['Test ACC']
            #results[1][fold] = summary['Best Val ACC']
            #results[2][fold] = summary['Test Loss']
            #results[3][fold] = summary['Best Val Loss']
            #results[4][fold] = summary['Test BPD']
            #results[5][fold] = summary['Best Val BPD']
    #print(results)
    result_means = np.mean(results, axis=-1)
    result_stds  = np.std( results, axis=-1)
    #print(result_means)
    #print(result_stds)
    for metric in range(metrics):
        print(r"{:.1f}\tiny$\pm${:.1f} & ".format(result_means[metric], result_stds[metric]))

if __name__ == '__main__':
    c = get_args()
    main(c)
