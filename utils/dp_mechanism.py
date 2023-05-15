import numpy as np

def calculate_noise_scale(args, times):

    # print (args.dp_epsilon)
    
    if args.dp_mechanism == 'Laplace':
        epsilon_single_query = args.dp_epsilon / times
        scale = 1
        return 1 / (scale * epsilon_single_query )

def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size