# pylint: disable=unused-import
import random
import time
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
import argparse
import sys
import torch
import dgl
import os
import time
import module
from utils import load_data, load_fixed_data_split
from utils import set_device, seed_everything
from utils import setup_cfg, tab_printer
from sklearn.metrics import accuracy_score as ACC

def print_res(res_list, info, matrix=False):
    if matrix:
        k = len(res_list)
        res = res_list[0]
        for i in range(k-1):
            res = res + res_list[i+1]
        print(info+':')
        print(res / k)
        return res / k
    
    else: 
        mean = np.mean(res_list)
        std = np.std(res_list)
        print(f'{info} mean ± std: ', '{:.2f} ± {:.2f}'.format(mean * 100, std * 100))
        return mean * 100


def main():
    device = set_device(args.device)
    graph, label, class_num = load_data(
        dataset_name=args.dataset,
        normalize = args.normalize,
        undirected=args.undirected,
        self_loop=args.self_loop,
    )
    if args.model_type in ['GloGNN', 'ACMGCN', 'CMGNN']:
        args.nnodes = graph.num_nodes()
    elif args.model_type == 'HOGGCN':
        adj = graph.adj(scipy_fmt='csr')
        args.adj = torch.tensor(adj.todense(),  dtype=torch.float32)

    t_start = time.time()
    seed_everything(args.seed)
    seed_list = [random.randint(0,99999) for i in range(args.runs * 100)]
    all_mask = torch.BoolTensor(torch.ones(graph.num_nodes(), dtype=bool))

    res_list_acc = []
    for run in range(args.runs):
        print(f"\nRun: {run}\n")
        seed_everything(seed_list[run])
        train_mask, val_mask, test_mask = load_fixed_data_split(args.dataset, run)

        Model = getattr(module, args.model_type)
        model = Model(
            in_features=graph.ndata['feat'].shape[1],
            class_num=class_num,
            device=device,
            args=args,
        )
        model.fit(
            graph,
            label,
            train_mask,
            val_mask,
            test_mask,
        )
        res, C, Z = model.predict(graph)
        acc = ACC(label[test_mask], res[test_mask])
        res_list_acc.append(acc)
        print(f"Acc: {acc}")    


    t_finish = time.time()
    print("Train cost: {:.4f}s".format(t_finish - t_start))
    print("\nResults:")
    acc_avg = print_res(res_list_acc, 'ACC')



def set_args(args, model_type):
    args.model_type = model_type
    if args.model_type == 'GloGNN':
        args.normalize = 1
        args.nhid = 64
        args.orders_func_id = 2
    elif args.model_type == 'GCNII':
        args.normalize = 1
        args.nhidden = 64
    elif args.model_type == 'GGCN':
        args.use_degree = True
        args.use_sign = True
        args.use_decay = True
        args.use_bn = False
        args.use_ln = False
        args.exponent = 3.0
        args.scale_init = 0.5
        args.deg_intercept_init = 0.5
    elif args.model_type == 'ACMGCN':
        args.nhid = 64
        args.nlayers = 1
        args.init_layers_X = 1
        args.acm_model = 'acmgcnp'
    elif args.model_type == 'OrderedGNN':
        args.chunk_size = 64
        args.hidden_channel = 256
        args.num_layers = 8
        args.add_self_loops = False
        args.simple_gating = False
        args.tm = True
        args.diff_or = True
    elif args.model_type == 'GPRGNN':
        args.init = 'PPR'
        args.dropout = 0.5
        args.nhidden = 64
        args.gamma = None
    elif args.model_type == 'GBKGNN':
        args.dim_size = 16
    elif args.model_type == 'HOGGCN':
        args.nhid1 = 512
        args.nhid2 = 256
        args.dropout = 0.5
    elif args.model_type == 'H2GCN':
        args.k = 2
        args.hidden_dim = 64
    elif args.model_type == 'MixHop':
        args.layers_1 = [200, 200, 200]
        args.dropout = 0.5
        args.layers_2 = [200, 200, 200]
        args.lambd = 0.0005
        args.cut_off = 0.1
        args.budget =60
    elif args.model_type in ['MLP', 'GCN', 'GAT']:
        pass
    elif args.model_type == 'CMGNN':
        args.self_loop = False
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='All', description='Parameters for Baseline Method')
    # experiment parameter
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Defaults to 4096.')
    parser.add_argument('--config', type=str, default='./config/xxxx.yaml', help='path of config file')
    parser.add_argument('--device', type=str, default='0', help='GPU id')
    parser.add_argument('--runs', type=int, default=10, help='The number of runs of task with same parmeter')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset used in the experiment')
    
    # train parameter 
    parser.add_argument('--epochs', type=int, default=2000, help='num of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=200, help='early stop patience')
    
    # model parameter
    parser.add_argument('--nhidden', type=int, default=128, help='num of hidden dimension')
    parser.add_argument('--undirected', type=bool, default=True, help='change graph to undirected')
    parser.add_argument('--self_loop', type=bool, default=True, help='add self loop')
    parser.add_argument('--normalize', type=int, default=-1, help='feature norm, -1 for without normalize')
    parser.add_argument('--layers', type=int, default=1, help='layers of model')
    parser.add_argument('--model_type', type=str, default='MLP', help='')

    args = parser.parse_args()
    args_dict = args.__dict__
   
    model_type = args_dict['model_type']
    if model_type == 'CMGNN':
        args_dict['config'] = f'./config/CMGNN.yaml'
    else:
        args_dict['config'] = f'./config/baseline/{model_type}.yaml'
    args = setup_cfg(args, args_dict)
    set_args(args, model_type)

    tab_printer(args_dict)
    main()


