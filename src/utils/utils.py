# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import numpy as np
import torch
import importlib
import datetime
import random


def format_table(metrics_dict):
    # Extract unique k values
    k_values = sorted(set(int(k.split('@')[1]) for k in metrics_dict.keys()))

    # Extract unique metric names
    metric_names = sorted(set(k.split('@')[0] for k in metrics_dict.keys()))

    # Prepare header
    header = ['@k'] + [metric[:6] for metric in metric_names]
    result = ['\t'.join(header)]

    # Prepare each row
    for k in k_values:
        row = [str(k)]
        for metric in metric_names:
            key = f"{metric}@{k}"
            row.append(f"{metrics_dict.get(key, np.nan):.4f}")
        result.append('\t'.join(row))

    return '\n'+'\n'.join(result) + '\n'


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """

    model_name_dict = {

        'proxy': 'Proxy',
        'random': 'Random',
        'popularity': 'Popularity',
        'bpr': 'BPR',
        'lgcn': 'LGCN',

        'kdvbpr': 'KDVBPR',
        'kdmmcf': 'KDMMCF',
        'kdlgcn': 'KDLGCN',


        'itemknncbf': 'ItemKNNCBF',
        'jpq_mm_itemknncbf': 'JPQ_MM_ITEMKNNCBF',

        'lightgcn': 'LightGCN',
        'slmrec': 'SLMRec',
        'vbpr': 'VBPR',
        'lattice': 'LATTICE',
        'mmgcn': 'MMGCN',
        'mmgcl': 'MMGCL',
        'freedom': 'FREEDOM',
        'mmgcf': 'MMGCF',

        'jpq_bpr': 'JPQ_BPR',
        'jpq_slmrec': 'JPQ_SLMREC',
        'jpq_vbpr': 'JPQ_VBPR',
        'jpq_lattice': 'JPQ_LATTICE',
        'jpq_mmgcn': 'JPQ_MMGCN',
        'jpq_mmgcl': 'JPQ_MMGCL',

        'jpq_mm_slmrec': 'JPQ_MM_SLMREC',
        'jpq_mm_vbpr': 'JPQ_MM_VBPR',
        'jpq_mm_lattice': 'JPQ_MM_SLMREC',
        'jpq_mm_mmgcn': 'JPQ_MM_MMGCN',
        'jpq_mm_mmgcl': 'JPQ_MM_MMGCL',

    }

    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if 'jpq' in model_file_name:
        module_path = '.'.join(['jpq_models', model_file_name])

    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_name = model_name_dict[model_name.lower()]

    # model_file_name = model_name.lower()
    # print(model_file_name)
    # module_path = '.'.join(['models', model_file_name])
    # if 'jpq' in model_file_name:
    #     module_path = '.'.join(['jpq_models', model_file_name])

    # if '_ete' in model_name:
    #     model_name = f"{model_name.split('_')[0].upper()}_ete"
    # else:
    #     model_name = model_name.upper()

    model_class = getattr(model_module, model_name)
    print(model_class)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return format_table(result_dict)


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(
        adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim
