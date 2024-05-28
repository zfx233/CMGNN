"""
common utils
"""
import csv
import os
import pickle as pkl
import random
import subprocess
from pathlib import Path
from pathlib import PurePath
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from texttable import Texttable
from torch import nn
import torch.nn.functional as F


def row_normalized_adjacency(adj, return_deg=False):
    adj = sp.coo_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum=(row_sum==0)*1+row_sum
    adj_normalized = adj/row_sum
    if return_deg:
        return sp.coo_matrix(adj_normalized), row_sum
    return sp.coo_matrix(adj_normalized)
   
def sys_normalized_adjacency(adj, return_deg=False):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    if return_deg:
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), rowsum
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor

    Args:
        sparse_mx (<class 'scipy.sparse'>): sparse matrix

    Returns:
        (torch.Tensor): torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def seed_everything(seed=42):
    """set random seed

    Parameters
    ----------
    seed : int
        random seed
    """
    # basic
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # dgl
    dgl.seed(seed)
    dgl.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def set_device(gpu: str = '0') -> torch.device:
    """Set torch device.

    Args:
        gpu (str): args.gpu. Defaults to '0'.

    Returns:
        torch.device: torch device. `device(type='cuda: x')` or `device(type='cpu')`.
    """
    max_device = torch.cuda.device_count() - 1
    if gpu == 'none':
        print('Use CPU.')
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        if not gpu.isnumeric():
            raise ValueError(
                f"args.gpu:{gpu} is not a single number for gpu setting."
                f"Multiple GPUs parallelism is not supported.")

        if int(gpu) <= max_device:
            print(f'GPU available. Use cuda:{gpu}.')
            device = torch.device(f'cuda:{gpu}')
            torch.cuda.set_device(device)
        else:
            print(
                f"cuda:{gpu} is not in available devices [0, {max_device}]. Use CPU instead."
            )
            device = torch.device('cpu')
    else:
        print("GPU is not available. Use CPU instead.")
        device = torch.device('cpu')
    return device
