import random
import time
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.options import args_parser
from models.Update import LocalUpdateDP, rnnUpdateDP
from models.Nets import CNNMnist, RNN
from models.Fed import FedWeightAvg
from models.test import test_img, test_name
# from opacus.grad_sample import GradSampleModule

if __name__ == '__main__':

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None
    net_glob = None

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        num_items = int(len(dataset_train) / args.num_users)
        all_idxs = [i for i in range(len(dataset_train))]
        for i in range(args.num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        net_glob = CNNMnist(args=args).to(args.device)
        # net_glob = GradSampleModule(net_glob)
    
    else:
        from utils.rnn_loader import RNNDataset

        n_letters = 57
        n_hidden = 128
        n_categories = 4 # Max is 18
        num_train_examples = 15000
        num_test_examples = 100
        args.lr = 0.005
        args.minibatch = 10

        dataset_train = RNNDataset(num_train_examples,True,n_categories)
        dataset_test= RNNDataset(num_test_examples,False,n_categories)
        num_items = int(num_train_examples / args.num_users)
        all_idxs = [i for i in range(num_train_examples)]
        for i in range(args.num_users):

            # This is for splitting into 2 languages per user to make data more non-IID
            if args.iid == 'false':
                sizeOfLanguageRange = num_train_examples // n_categories
                languageOne = random.randint(0, n_categories - 1)
                languageTwo = ( languageOne + 1 ) % n_categories
                # languageThree = ( languageTwo + 1 ) % n_categories
                l1 = all_idxs[languageOne * sizeOfLanguageRange : (languageOne * sizeOfLanguageRange) + sizeOfLanguageRange]
                l2 = all_idxs[languageTwo * sizeOfLanguageRange : (languageTwo * sizeOfLanguageRange) + sizeOfLanguageRange]
                new_idxs = l1 + l2
                dict_users[i] = set(np.random.choice(new_idxs, num_items, replace=False))
            else:
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
        net_glob = RNN(n_letters, n_hidden, n_categories).to(args.device)

    img_size = dataset_train[0][0].shape

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # training
    acc_test = []
    if args.dataset == 'mnist':
        clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    else:
        clients = [rnnUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)

    for iter in range(args.epochs):
        t_start = time.time()
        w_locals, loss_locals, weight_locols = [], [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]

        max_idxs_size = 1e-6
        min_lr = args.lr
        for idx in idxs_users:
            local = clients[idx]
            # RANDOM SAMPLE AKA Batch
            random_sample_size = len(local.idxs_sample)
            new_lr = local.lr
            if  random_sample_size > max_idxs_size:
                max_idxs_size = random_sample_size
            if new_lr < min_lr:
                min_lr = new_lr
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # print(loss)
            # quit()
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        # update global weights
        w_glob = FedWeightAvg(w_locals, weight_locols, args, max_idxs_size, new_lr)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        if args.dataset == 'mnist':
            acc_t, loss_t = test_img(net_glob, dataset_test, args)
        else:
            acc_t, loss_t = test_name(net_glob, dataset_test, args)

        t_end = time.time()
        print("{:3d},{:.2f}".format(iter, acc_t))