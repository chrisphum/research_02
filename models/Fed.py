#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from utils.dp_mechanism import cal_sensitivity, calculate_noise_scale
import numpy as np

def FedWeightAvg(w, size, args, max_idxs_size, lr):

    times = args.epochs * args.frac
    totalSize = sum(size)

    if args.dp_epsilon_global > 0:
        noise_scale = calculate_noise_scale(args, times)
        sensitivity = cal_sensitivity(lr, args.dp_clip, max_idxs_size)
    
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        w_avg[k] = torch.div(w_avg[k], totalSize)
        if args.dp_mechanism == 'Laplace' and args.dp_epsilon_global > 0:
            w_avg[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * noise_scale,size=w_avg[k].shape))

    return w_avg