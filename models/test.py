#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def test_name(net_g, datatest, args):
    net_g.eval()
    test_loss=0
    correct=0
    numSamples = 0
    for line_tensor, target in datatest:
        hidden = torch.zeros(1, 128)
        log_probs = None
        for i in range(line_tensor.size()[0]):
            log_probs, hidden = net_g(line_tensor[i], hidden)
            # log_probs = net_g(line_tensor[i])

        # print(log_probs)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        numSamples += 1

    accuracy = 100.00 * correct / numSamples
    test_loss /= numSamples
    return accuracy, test_loss

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=1024)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available() and args.gpu != -1:
            data, target = data.cuda(args.device), target.cuda(args.device)
        else:
            data, target = data.cpu(), target.cpu()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

