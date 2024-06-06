import os
import numpy as np
import torch
import torch.nn as nn
from time import time
from sklearn.metrics import accuracy_score
from utils.helpers import t2np, StatsRecorder

try: import wandb
except: pass

log_theta = nn.LogSigmoid()  # nn.Identity()
theta = nn.Sigmoid()

default_config = {
        'log_timing': False,
        'max_eval_ex': float('inf'),
        'warmup_epochs': 4,
        'notes': 'classification',
    }

class ClExperiment:
    def __init__(self, size, model, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.config = default_config
        self.config.update(**kwargs)
        self.verbose = self.config['verbose']
        self.summary = {}
        if self.verbose: print(self.model)

        if self.config['wandb']:
            wandb.init(name=self.config['name'],
                       notes=self.config['notes'],
                       project=self.config['wandb_project'], 
                       entity=self.config['wandb_entity'], 
                       config=self.config)
            wandb.watch(self.model)

        self.update_summary('Epoch', 0)
        self.update_summary('Best Val ACC', 0.0)
        # latency
        if self.config['log_timing']:
            self.train_batch_time = StatsRecorder()
            self.eval_batch_time  = StatsRecorder()
            self.sample_time = StatsRecorder()

        self.device = self.config['device']
        self.dim_inv = 1.0 / torch.prod(torch.tensor(size))
        self.alpha = 1e-3 if self.criterion else 1e-0

    def run(self):
        for e in range(self.summary['Epoch'] + 1, self.config['epochs'] + 1):
            self.update_summary('Epoch', e)
            if e > self.config['eval_epochs']:
                train_loss = self.train_epoch(e)
                self.log('Train Loss', train_loss)
                self.scheduler.step()

            if e % self.config['eval_epochs'] == self.config['eval_epochs']-1:
                valid_loss, valid_logpx, valid_gt_list, valid_sc_list = self.eval_epoch(self.valid_loader, e, split='Val')
                valid_acc = 1e2*accuracy_score(np.asarray(valid_gt_list), np.asarray(valid_sc_list))
                # update stats
                self.log('Val Loss', valid_loss)
                self.log('Val ACC' , valid_acc)
                if valid_acc > self.summary['Best Val ACC']:
                    self.update_summary('Best Val Loss', valid_loss)
                    self.update_summary('Best Val ACC' , valid_acc)
                    self.update_summary('Best Val Epoch', e)
                    test_loss, test_logpx, test_gt_list, test_sc_list = self.eval_epoch(self.test_loader, e, split='Test')
                    test_acc = 1e2*accuracy_score(np.asarray(test_gt_list), np.asarray(test_sc_list))
                    self.log('Test Loss', test_loss)
                    self.log('Test ACC' , test_acc)
                    self.update_summary('Test Loss', test_loss)
                    self.update_summary('Test ACC' , test_acc)
                    # checkpoint model
                    if self.config['save_checkpoint']: self.save()

    def log(self, name, val):
        if self.verbose:
            if (isinstance(val, str) or isinstance(val, int)): print("{}: {}".format(name, val))
            else: print("{}: {:.2f}".format(name, val))
        if self.config['wandb']: wandb.log({name: val})

    def update_summary(self, name, val):
        if self.verbose:
            if (isinstance(val, str) or isinstance(val, int)): print("{}: {}".format(name, val))
            else: print("{}: {:.2f}".format(name, val))
        self.summary[name] = val if (isinstance(val, str) or isinstance(val, int)) else round(val*100)/100.0
        if self.config['wandb']: wandb.run.summary[name] = val

    def warmup_lr(self, epoch, num_batches):
        if epoch <= self.config['warmup_epochs']:
            for param_group in self.optimizer.param_groups:
                s = (((num_batches+1) + (epoch-1) * len(self.train_loader)) 
                        / (self.config['warmup_epochs'] * len(self.train_loader)))
                param_group['lr'] = self.config['lr'] * s
        for param_group in self.optimizer.param_groups: lr = param_group['lr']
        return lr

    def train_epoch(self, epoch):
        loss_sup = 0.0
        loss_uns = 0.0
        loss_sum = 0.0
        count = 0
        batch_durations = []
        self.model.train()
        for x, gt, c in self.train_loader:
            lr = self.warmup_lr(epoch, count)
            self.optimizer.zero_grad()
            # label/data:
            x  = x.to(self.device)
            gt = gt.to(self.device)
            context = c.to(self.device)
            # latency:
            if self.config['log_timing']:
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
            # model:
            logp = self.dim_inv*self.model.log_prob(x, context=context)
            logp[logp != logp] = 0.0  # replace NaN's with 0
            # loss:
            cost_uns =-self.alpha*log_theta(torch.logsumexp(logp, -1)).mean()
            if self.criterion: cost_sup = self.criterion(logp, gt)
            else: cost_sup = torch.zeros_like(cost_uns)
            cost_sum = cost_sup + cost_uns
            cost_sum.backward()
            if self.config['grad_clip_norm'] is not None: nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            self.optimizer.step()
            # latency:
            if self.config['log_timing']:
                end.record()
                torch.cuda.synchronize()
                batch_durations.append(start.elapsed_time(end))
            # loss:
            loss_sup += cost_sup.item()
            loss_uns += cost_uns.item()
            loss_sum += cost_sum.item()
            count += 1
            if count % self.config['log_interval'] == 0:
                self.log('Train Batch Loss', loss_sum / count)

        if self.config['log_timing']:
            # take all but first 10 and last 10 batch times into account
            self.train_batch_time.update(batch_durations[10:-10])
            self.update_summary('Train Batch Time Mean', self.train_batch_time.mean)
            self.update_summary('Train Batch Time Std',  self.train_batch_time.std)

        mean_loss_sup = loss_sup / count
        mean_loss_uns = loss_uns / count
        mean_loss_sum = loss_sum / count
        print('Epoch: {:d} train loss: {:.4f}={:.4f}+{:.4f}, lr={:.6f}'.format(
            epoch, mean_loss_sum, mean_loss_sup, mean_loss_uns, lr))
        return mean_loss_sum

    def eval_epoch(self, loader, epoch, split='Val'):
        gt_list = list()
        sc_list = list()
        loss_sup = 0.0
        loss_uns = 0.0
        loss_sum = 0.0
        log_px   = 0.0
        count = 0
        batch_durations = []
        with torch.no_grad():
            self.model.eval()
            for x, gt, c in loader:
                # label/data:
                x  = x.to(self.device)
                gt = gt.to(self.device)
                context = c.to(self.device)
                # latency:
                if self.config['log_timing']:
                    start = torch.cuda.Event(enable_timing=True)
                    end   = torch.cuda.Event(enable_timing=True)
                    start.record()
                # model:
                logp = self.dim_inv*self.model.log_prob(x, context=context)
                logp[logp != logp] = 0.0  # replace NaN's with 0
                # latency:
                if self.config['log_timing']:
                    end.record()
                    torch.cuda.synchronize()
                    batch_durations.append(start.elapsed_time(end))
                # loss:
                cost_uns =-self.alpha*log_theta(torch.logsumexp(logp, -1)).mean()
                if self.criterion: cost_sup = self.criterion(logp, gt)
                else: cost_sup = torch.zeros_like(cost_uns)
                cost_sum = cost_sup + cost_uns
                loss_sup += cost_sup.item()
                loss_uns += cost_uns.item()
                loss_sum += cost_sum.item()
                log_px   += torch.logsumexp(logp, -1).sum().item()
                count += 1
                sc = torch.argmax(logp, dim=-1)
                gt_list.extend(t2np(gt))
                sc_list.extend(t2np(sc))
                if count >= self.config['max_eval_ex']: break
        
            if self.config['log_timing']:
                # take all but first 10 and last 10 batch times into account
                self.eval_batch_time.update(batch_durations[10:])
                self.update_summary('Eval Batch Time Mean', self.eval_batch_time.mean)
                self.update_summary('Eval Batch Time Std',  self.eval_batch_time.std)

        mean_loss_sum = loss_sum / count
        mean_log_px   = log_px   / count
        return mean_loss_sum, mean_log_px, gt_list, sc_list

    def save(self):
        self.log('Note', f'Saving checkpoint to: {self.config["checkpoint_path"]}')
        checkpoint = {
                      'summary': self.summary,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_state_dict': self.scheduler.state_dict(),
                      'config': self.config}

        torch.save(checkpoint, self.config['checkpoint_path'])

    def load(self, path):
        self.log('Note', f'Loading checkpoint from: {path}')
        checkpoint = torch.load(path)

        # warning, config params overwritten
        if 'generalist' in path:
            generalist_summary = checkpoint['summary']
            print('Generalist summary:', generalist_summary)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            assert len(unexpected_keys) == 0, 'generalist unexpected_keys list should be empty'
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)
        else:
            self.summary = checkpoint['summary']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        config_diff = set(self.config.items()) ^ set(checkpoint['config'].items())
        if config_diff != set(): self.log('Warning', f'Differences in loaded config: {config_diff}')
