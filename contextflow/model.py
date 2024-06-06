import os, time, math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from experiment_ad import AdExperiment
from experiment_cl import ClExperiment
from datasets.mnist import load_data as load_data_mnist
from datasets.cifar10 import load_data as load_data_cifar10
from datasets.ts import load_data_ts
from config import get_args
from utils.helpers import init_seeds
from layers.rtdl.nn._embeddings import *
from layers import *
from utils.tranad_pot import pot_eval
from datasets.mtad_dataloader import mtad_entities
from torchinfo import summary
from ood_metrics import auroc, aupr, fpr_at_95_tpr
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
                            balanced_accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, precision_recall_curve)

activations = {
    'SLR':    lambda size: SmoothLeakyRelu(alpha=0.3),
    'Spline': lambda size: SplineActivation(size, tail_bound=10, individual_weights=True),
    'LLR':    lambda size: LearnableLeakyRelu(),
}


class ContextEncoder(nn.Sequential):
    def __init__(self, contexts, enc_emb, enc_type, data_size, init='orthogonal'):
        if   enc_emb == 'onehot': # and init != 'dist':
            sz = (sum(contexts),)  # one-hot
            emb = OneHotEncoder(contexts)
            num_cats = sum(contexts)*[1]  #
        elif enc_emb == 'eye' and enc_type == 'argmax':
            sz = (sum(ArgmaxCatDequantization.cats2bits(contexts)),)
            sz = (sz[0]+sz[0]%2, )
            emb = EyeEncoder()
            num_cats = contexts  #
        elif enc_emb == 'eye' and enc_type != 'argmax':
            sz = (len(contexts),)  #
            emb = EyeEncoder()
            num_cats = contexts  #
        elif enc_emb == 'embed':
            emb = CatEmbeddings(contexts, data_size[0], stack=False, init=init)
            sz = (data_size[0]*len(contexts),)  # embedding
        else:
            raise NotImplementedError('{} is not supported enc-emb!'.format(enc_emb))            
        
        layers = []
        if enc_type in ['vardeq', 'argmax', 'probsample']:
            num_layers = 2
            linear = True  # True False
            actnorm = True  # True False
            coupling = True  # True False
            activation = False  # True False
            act = activations['Spline']
            
            for _ in range(num_layers):
                if sz[0] % 2:
                    layers.append(Augment(StandardNormal((1,)), 1))
                    sz = (sz[0]+1,)
                
                if linear: layers.append(FC(sz))
                if actnorm: layers.append(ActNormFC(sz))
                if activation: layers.append(act(sz))
                if coupling: layers.append(CouplingFC(sz[0]))
        
        if   enc_type == 'eyesample':
            encoder = EyeSampling()
        elif enc_type == 'probsample':
            encoder = ProbSampling(FlowInvSequential(
                ConditionalGaussianDistribution(size=sz, context_net=CatEmbeddings(contexts, 2*sz[0]//len(contexts), stack=False, init='zeros')), *layers))
        elif enc_type == 'uniform':
            encoder = UniformCatDequantization(num_cats=num_cats)
        elif enc_type == 'vardeq':
            context_net = CatEmbeddings(contexts, 2*sz[0]//len(contexts), stack=False, init='zeros')
            encoder = VariationalCatDequantization(FlowInvSequential(
                ConditionalGaussianDistribution(size=sz, context_net=context_net), *layers), num_cats=num_cats)
        elif enc_type == 'argmax':
            context_net = CatEmbeddings(contexts, 2*sz[0]//len(contexts), stack=False, init='zeros')
            encoder = ArgmaxCatDequantization(FlowInvSequential(
                ConditionalGaussianDistribution(size=sz, context_net=context_net), *layers), num_cats=num_cats)
        else:
            raise NotImplementedError('{} is not supported enc-type!'.format(enc_type))            
        
        self.C = sz[0]
        self.contexts = contexts
        super(ContextEncoder, self).__init__(emb, encoder)


kwargs_context = {'eye': lambda contexts, enc_emb, enc_type, data_size, init='orthogonal': None, 'ours': ContextEncoder}

def create_model(config, data_size=(1, 1, 1), mixtures=1, contexts=[-1]):
    alpha = 1e-4
    if config['dataset'] in ['mnist', 'cifar10']:
        layers = [Dequantization(UniformDistribution(size=data_size)),  # [0:255] + noise -> [0:1] -> logit([0:1])
                  Normalization(translation = 0.0, scale = 256.0), Normalization(translation = alpha, scale = 1/(1-2*alpha)), 
                  LogitTransform()]
    else: layers = []
    #
    contextflow = config['contextflow']
    generalist  = config['generalist']
    #act = activations[config['activation']]
    #
    if mixtures == 1:               dist = GaussianMixtureDistribution  # StandardNormal
    elif config['dist'] == 'gauss': dist = GaussianMixtureDistribution
    elif config['dist'] == 'tdist': dist = StudentMixtureDistribution
    else: raise NotImplementedError('{} is not supported base distribution!'.format(config['dist']))
    #
    sz = data_size
    ts_dataset = True if c.dataset in ['atm', 'msl', 'smd', 'smap'] else False
    p, krn, pad = ((2,1), (3,1), (1,0)) if ts_dataset else ((2,2), (3,3), (1,1))
    components = 8  # number of components in the GMM
    # context
    mode = 'eye' if generalist else 'ours'
    enc_emb  = config['enc_emb']
    enc_type = config['enc_type']
    for l in range(config['num_blocks']):
        if sz[0] % 2:
            layers.append(Augment(StandardNormal((1, sz[1], sz[2])), 1))
            sz = (sz[0]+1, sz[1], sz[2])
        
        if c.dataset not in ['msl', 'smd', 'smap']:
            layers.append(Squeeze(patch_size=p))
            sz = (sz[0]*p[0]*p[1], sz[1]//p[0], sz[2]//p[1])
        
        for k in range(config['block_size']):
            context_net = kwargs_context[mode](contexts, enc_emb, enc_type, (sz[0],), init='zeros')  # (sz[0],)
            layers.append(Conv1x1(sz, context_net=context_net, contextflow=contextflow))

            if config['actnorm']:
                context_net = kwargs_context[mode](contexts, enc_emb, enc_type, (2*sz[0],))
                layers.append(ActNorm(sz, context_net=context_net, contextflow=contextflow))
            
            #if not (l == config['num_blocks']-1 and k == config['block_size']-1): layers.append(act(sz))

            if config['coupling'] == 'trans' and sz[1] % p[0] == 0 and sz[2] % p[1] == 0:
                context_net = kwargs_context[mode](contexts, enc_emb, enc_type, (sz[0],))
                layers.append(TransCoupling(sz, p, context_net=context_net, contextflow=contextflow))
            elif config['coupling'] == 'conv':
                context_net = kwargs_context[mode](contexts, enc_emb, enc_type, (sz[0],))
                layers.append(Coupling(sz[0], kernel_size=krn, padding=pad, context_net=context_net, contextflow=contextflow))
            elif config['coupling'] == 'maf':
                context_net = kwargs_context[mode](contexts, enc_emb, enc_type, (sz[0],))
                layers.append(MaskedCoupling(sz[0], kernel_size=krn, padding=pad, context_net=context_net, contextflow=contextflow))
            
            if config['dataset'] in ['atm']:  # ts_dataset:
                layers.append(PermuteAxes((0,2,1,3)))
                sz = (sz[1], sz[0], sz[2])

        if config['split_prior'] and l < config['num_blocks']-1:
            sz = (sz[0]//2, sz[1], sz[2])
            #layers.append(SplitPrior(dist(size=sz, mixtures=mixtures)))
            #context_net = kwargs_context[mode](contexts, enc_emb, enc_type, (2*mixtures*sz[0],), init='dist')
            context_net = kwargs_context[mode](contexts, 'embed', 'eyesample', (2*mixtures*components*sz[0]//len(contexts),), init='zeros')  # for speed we use a simple lookup
            layers.append(SplitPrior(dist(size=sz, mixtures=mixtures, components=components, context_net=context_net, contextflow=contextflow)))

    #return FlowSequential(dist(size=sz, mixtures=mixtures), *layers)
    #context_net = kwargs_context[mode](contexts, enc_emb, enc_type, (2*mixtures*sz[0],), init='dist')
    context_net = kwargs_context[mode](contexts, 'embed', 'eyesample', (2*mixtures*components*sz[0]//len(contexts),), init='zeros')  # for speed we use a simple lookup
    return FlowSequential(dist(size=sz, mixtures=mixtures, components=components, context_net=context_net, contextflow=contextflow), *layers)


def main(c):
    # device:
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    c.device = torch.device("cuda:{}".format(c.gpu) if c.use_cuda else "cpu")
    generalist = True if 'generalist' in c.action_type else False
    # model:
    if c.dataset == 'mnist':
        window_length = 32
        mixtures = 10
        num_blocks = 2
        block_size = 2
        actnorm = True
        split_prior = False
        contexts = [-1] if c.clean else [64]
    elif c.dataset == 'cifar10':
        window_length = 32
        mixtures = 10
        num_blocks = 3
        actnorm = True
        block_size = 4
        split_prior = True
        contexts = [-1, -1] if c.clean else [15, 5]
    elif c.dataset == 'atm':
        window_length = 144
        c.preprocessing = 'minmax'
        num_blocks = 3
        block_size = 4
        actnorm = True
        split_prior = True
        contexts = [68]  # discrete contexts
        data_vars = (26, 6+6)  # (continuous, discrete+zero)
        mixtures = 2
    elif c.dataset in ['msl', 'smd', 'smap']:
        # debug:
        #mixtures = 2
        #c.supervision = 'full'
        # benchmark:
        c.supervision = 'weak'
        c.preprocessing = 'minmax'
        window_length = 8
        mixtures = 1
        num_blocks = 2
        block_size = 4
        actnorm = True  # False
        split_prior = False  # True
        if c.dataset == 'msl':
            contexts = [27]  # discrete contexts
            data_vars = (55, 0)
        elif c.dataset == 'smd':
            contexts = [28]  # discrete contexts
            data_vars = (38, 0)
        elif c.dataset == 'smap':
            contexts = [55]  # discrete contexts
            data_vars = (25, 0)
            
    else: raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    ad_dataset = True if c.dataset in ['atm', 'msl', 'smd', 'smap'] else False
    # upd config:
    if generalist:
        context = [-1] if c.clean else contexts
        postfix = 'generalist_clean' if c.clean else 'generalist'
    else:
        context = contexts
        context_type = 'contextflow' if c.contextflow else 'conventional'
        postfix = 'specialist_{}_{}_{}_{}'.format(contexts, c.enc_emb, c.enc_type, context_type)
    
    run_name = '{}_{}_{}L_{}B_{}_{}_F{}'.format(c.dataset, c.supervision, num_blocks, block_size, c.coupling, c.dist, c.fold)
    if not os.path.exists('../checkpoints'): os.makedirs('../checkpoints')
    checkpoint = c.checkpoint
    generalist_path = os.path.join('../checkpoints', '{}_{}.pt'.format(run_name, 'generalist'))
    checkpoint_path = os.path.join('../checkpoints', '{}_{}.pt'.format(run_name, postfix))
    print('checkpoint path:', checkpoint_path)
    config = {
        'epochs': c.epochs,
        'device': c.device,
        'dataset': c.dataset,
        'name': '{}_{}'.format(run_name, postfix),
        'checkpoint_path': checkpoint_path,
        'generalist_path': generalist_path,
        'save_checkpoint': c.save_checkpoint,
        'wandb': c.wandb,
        'wandb_project': c.dataset,
        'wandb_entity': 'YOUR_ID',
        'verbose': c.verbose,
        'eval_epochs': 1,
        'log_interval': float('inf'),
        'lr': c.lr,
        'num_blocks': num_blocks,
        'block_size': block_size,
        'contextflow': c.contextflow,
        'generalist': generalist,
        'enc_emb': c.enc_emb,
        'enc_type': c.enc_type,
        'actnorm': actnorm,
        'split_prior': split_prior,
        'activation': 'Spline',
        'grad_clip_norm': None,
        'coupling': c.coupling,
        'window_length': window_length,
        'dist': c.dist
    }
    print(config)
    if c.dataset == 'mnist':
        train_loader, valid_loader, test_loader = load_data_mnist(c, eval_context=context, contexts=contexts)
        data_size = (1, window_length, window_length)
        train_weight = torch.ones(mixtures)
    elif c.dataset == 'cifar10':
        train_loader, valid_loader, test_loader = load_data_cifar10(c, eval_context=context, contexts=contexts)
        data_size = (3, window_length, window_length)
        train_weight = torch.ones(mixtures)
    elif c.dataset in ['atm', 'msl', 'smd', 'smap']:
        c.data_vars, c.window_length = data_vars, window_length
        #context = [0,1,2]  # C-1 for MSL
        (train_loader, valid_loader), (train_weight, _) = load_data_ts(c, context=context, contexts=contexts)
        data_size = (sum(data_vars), window_length, 1)
        test_loader = valid_loader
    else: raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))

    model = create_model(config, data_size=data_size, mixtures=mixtures, contexts=contexts).to(c.device)
    #  prints complexity estimates using torchinfo
    if c.complexity: summary(model, [(c.batch_size,) + data_size, (c.batch_size, len(contexts))], dtypes=[torch.float, torch.long])
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=c.epochs//4, gamma=0.1)
    
    if c.supervision == 'weak':
        criterion = None
    else: criterion = nn.CrossEntropyLoss(weight=train_weight.to(c.device))
    
    if ad_dataset: experiment = AdExperiment(data_size, model, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, **config)
    else:          experiment = ClExperiment(data_size, model, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, **config)
    #
    if checkpoint:
        experiment.load(checkpoint)
    elif c.action_type in ['test-specialist', 'test-generalist'] and os.path.isfile(checkpoint_path):
        print(config['checkpoint_path'])
        experiment.load(config['checkpoint_path'])
    elif c.action_type == 'train-specialist' and c.contextflow and os.path.isfile(generalist_path):
        experiment.load(config['generalist_path'])
    
    if c.action_type in ['test-specialist', 'test-generalist']:
        experiment.summary['Epoch'] = 0
        experiment.summary['Best Val ACC'] = 0.0      
        experiment.config['eval_epochs'] = 1
        experiment.config['epochs'] = 1
        experiment.run()
        print(experiment.summary)
    #if c.action_type in ['test-specialist', 'test-generalist']:
    #    mf1, mauc, mpr, mrec, cnt = 0.0, 0.0, 0.0, 0.0, 0.0
    #    for cur_context in range(contexts[0]):
    #        #print(cur_context, contexts)
    #        #print(mtad_entities[c.dataset])
    #        (train_loader, valid_loader), (train_weight, _) = load_data_ts(c, context=[cur_context], contexts=contexts)
    #        _, _            , train_sc_list, train_id_list = experiment.eval_epoch(train_loader, 0, split='Train')
    #        _, valid_gt_list, valid_sc_list, valid_id_list = experiment.eval_epoch(valid_loader, 0, split='Val')
    #        train_ad_score = np.asarray(train_sc_list, dtype=float)
    #        valid_ad_score = np.asarray(valid_sc_list, dtype=float)
    #        valid_gt_label = np.asarray(valid_gt_list, dtype=int)
    #        #print(train_ad_score.shape, valid_ad_score.shape, valid_gt_label.shape)
    #        result, _ = pot_eval(c.dataset, train_ad_score, valid_ad_score, valid_gt_label)
    #        #print(result)
    #        f1  = 1e2*result['f1']
    #        auc = 1e2*result['ROC/AUC']
    #        pr  = 1e2*result['precision']
    #        rec = 1e2*result['recall']
    #        mf1 += f1
    #        mauc+= auc
    #        mpr += pr
    #        mrec+= rec
    #        cnt += 1
    #        print('{} Test P/R/AUC/F1 {:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(mtad_entities[c.dataset][cur_context], pr, rec, auc, f1))
    #    print('MEAN Test P/R/AUC/F1 {:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(mpr/cnt, mrec/cnt, mauc/cnt, mf1/cnt))
#
        #valid_loss, valid_gt_list, valid_sc_list, valid_id_list = experiment.eval_epoch(test_loader, 0, split='Val')
        ## score aggregation
        #gt_score = np.asarray(valid_gt_list, dtype=float)
        #ad_score = np.asarray(valid_sc_list, dtype=float)
        #precision, recall, thresholds = precision_recall_curve(gt_score, ad_score)
        #a = 2 * precision * recall
        #b = precision + recall
        #f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        #threshold = thresholds[np.argmax(f1)]
        #threshold = min(threshold, 0.999)
        #gt_label = np.asarray(valid_gt_list, dtype=int)
        #ad_label = (ad_score > threshold).astype(int)
        #print('Optimal Threshold: {:.8f} with ad_score mean = {:.8f}'.format(threshold, np.mean(ad_score)))
        ## sklearn:
        #ac = 1e2*accuracy_score(         gt_label, ad_label)
        #f1 = 1e2*f1_score(               gt_label, ad_label)
        #auc= 1e2*auroc(                  ad_score, gt_score)
        #print('Test acc/auc/f1 {:.1f}/{:.1f}/{:.1f}'.format(ac, f1, auc))
        #print(experiment.summary)
    
        #test_loss, test_gt_list, test_sc_list, test_id_list = experiment.eval_epoch(test_loader, 0, split='Val')
        #for c1 in range(contexts[0]):
        #    for c2 in range(contexts[1]):
        #        eval_context = [c1, c2]
        #        train_loader, valid_loader, test_loader = load_data_cifar10(c, eval_context=eval_context, contexts=contexts)
        #        test_loss, test_gt_list, test_sc_list, test_id_list = experiment.eval_epoch(test_loader, 0, split='Val')
        #        test_acc = 1e2*accuracy_score(np.asarray(test_gt_list), np.asarray(test_sc_list))
        #        print('Test acc/loss {:.1f}/{:.2f} for {} context'.format(test_acc, test_loss, eval_context))
    elif c.action_type in ['train-specialist', 'train-generalist']:
        experiment.run()
        print(experiment.summary)
    else: raise NotImplementedError('{} is not supported action type!'.format(c.action_type))

if __name__ == '__main__':
    c = get_args()
    main(c)