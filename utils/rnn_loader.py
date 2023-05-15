from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import random
import unicodedata
import string
import torch.nn as nn
from torch.utils.data import Dataset

class RNNDataset(Dataset):

    def __init__(self, numberOfExamples, train,max_categories):

        def findFiles(path): return glob.glob(path)

        all_letters = string.ascii_letters + " .,;'"
        n_letters = len(all_letters)

        # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        def unicodeToAscii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
                and c in all_letters
            )

        # Build the category_lines dictionary, a list of names per language
        category_lines = {}
        all_categories = []
        

        # Read a file and split into lines
        def readLines(filename):
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            return [unicodeToAscii(line) for line in lines]

        for filename in findFiles('data/names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            lines = readLines(filename)
            category_lines[category] = lines
            if len(all_categories) + 1 > max_categories:
                break

        def letterToIndex(letter):
            return all_letters.find(letter)

        def lineToTensor(line):
            tensor = torch.zeros(len(line), 1, n_letters)
            for li, letter in enumerate(line):
                tensor[li][0][letterToIndex(letter)] = 1
            return tensor

        def randomChoice(l):
            return l[random.randint(0, len(l) - 1)]

        def randomTrainingExample(category):
            line = randomChoice(category_lines[category])
            category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
            line_tensor = lineToTensor(line)
            return category, line, category_tensor, line_tensor
        
        # print(n_letters)
        # n_categories = len(all_categories)
        # print(n_categories)

        self.train = train
        data_values = []
        data_labels = []

        for i in range(numberOfExamples):
            # category = randomChoice(all_categories)
            category = all_categories[int(i/(numberOfExamples // max_categories))]
            category, line, category_tensor, line_tensor = randomTrainingExample(category)
            # print('category =', category, '/ line =', line)
            data_values.append(line_tensor)
            data_labels.append(category_tensor)
        self.data = data_values
        self.label = data_labels

    def __getitem__(self, index):
        name, target = self.data[index], self.label[index]
        return name, target



# criterion = nn.NLLLoss()







# learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

# def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()

#     rnn.zero_grad()

#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)

#     loss = criterion(output, category_tensor)
#     loss.backward()

#     # Add parameters' gradients to their values, multiplied by learning rate
#     for p in rnn.parameters():
#         p.data.add_(p.grad.data, alpha=-learning_rate)

#     return output, loss.item()

# import time
# import math

# n_iters = 100000
# print_every = 5000
# plot_every = 1000



# # Keep track of losses for plotting
# current_loss = 0
# all_losses = []

# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)

# start = time.time()

# for iter in range(1, n_iters + 1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output, loss = train(category_tensor, line_tensor)
#     current_loss += loss

#     # Print ``iter`` number, loss, name and guess
#     if iter % print_every == 0:
#         guess, guess_i = categoryFromOutput(output)
#         correct = '✓' if guess == category else '✗ (%s)' % category
#         print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

#     # Add current loss avg to list of losses
#     if iter % plot_every == 0:
#         all_losses.append(current_loss / plot_every)
#         current_loss = 0




# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# plt.figure()
# plt.plot(all_losses)