#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, calculate_noise_scale
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class rnnUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None):

        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(len(idxs)), replace=False)
        self.ldr_train = DatasetSplit(dataset, self.idxs_sample)
        self.idxs = idxs
        self.times = self.args.epochs * self.args.frac
        
        self.lr = args.lr
        self.noise_scale = calculate_noise_scale(args.dp_epsilon,self.times)  # noise-scale = CE / epsilon   sensitivity = 2 * lr * clip / m , combined = (CE* 2* lr* clip*) / (epislon * m*)
        self.criterion = nn.NLLLoss()


    def train(self, net):
        net.train()
        loss_client = 0 
        count = 0
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        total_loss = 0
        net.zero_grad()
        for line_tensor, category_tensor in self.ldr_train:
                hidden = torch.zeros(1, 128)
                # net.zero_grad()
                for i in range(line_tensor.size()[0]):
                    output, hidden = net(line_tensor[i], hidden)
                loss = self.criterion(output, category_tensor)
                # loss = loss  / self.args.minibatch
                loss.backward()
                if self.args.dp_mechanism != 'no_dp':
                    self.clip_gradients(net)
                count += 1
                if (self.args.minibatch == 0) or (count % self.args.minibatch == 0) or (count == len(self.idxs_sample)):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if self.args.dp_mechanism != 'no_dp':
                        self.add_noise(net)
        loss_client = loss.item()
        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), loss_client
    
    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            for k, v in net.named_parameters():
                # print(v.grad.norm(1))
                v.grad /= max(1, v.grad.norm(1) / self.args.dp_clip)
                # print(v.grad.norm(1))

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))
        # print(sensitivity)
        state_dict = net.state_dict()
        if self.args.dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
                
        net.load_state_dict(state_dict)
    
class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None):

        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(len(idxs)), replace=False)
        if args.minibatch != 0:
            self.miniBS = args.minibatch
        else:
            self.miniBS = int(len(self.idxs_sample))
        self.repetitionsPerEpoch = int(len(self.idxs_sample)) // self.miniBS
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=self.miniBS,shuffle=True)
        self.idxs = idxs
        self.times = self.args.epochs * self.args.frac
        self.lr = args.lr
        self.noise_scale = calculate_noise_scale(args.dp_epsilon,self.times)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_client = 0
        for batchNum in range(self.repetitionsPerEpoch):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                if self.args.dp_mechanism != 'no_dp':
                    self.clip_gradients(net)
                optimizer.step()
                scheduler.step()
                # add noises to parameters
                if self.args.dp_mechanism != 'no_dp':
                    self.add_noise(net)
                loss_client = loss.item()
        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), loss_client
    
    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            self.per_sample_clip(net, self.args.dp_clip, norm=1)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))
        state_dict = net.state_dict()
        if self.args.dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
                
        net.load_state_dict(state_dict)
