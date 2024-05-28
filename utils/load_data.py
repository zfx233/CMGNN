from typing import Tuple
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dgl.data.utils import save_graphs, load_graphs

table_datasets = [
    'actor', 'chameleonf', 'squirrelf', 
    'romanempire', 'amazonratings', 
    'blogcatalog', 'flickr', 
    'photo',  'wikics', 'pubmed', 
]

def load_data(
    dataset_name: str,
    normalize: int = -1,
    undirected: bool=True,
    self_loop: bool=True,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    dataset_name = dataset_name.lower()
    if dataset_name in table_datasets:
        print("Load Dataset: ", dataset_name)
        file_path = f'data/graphs/{dataset_name}.pt'
        graphs, _ = load_graphs(file_path)
        graph = graphs[0]
        label = graph.ndata['label']
        class_num = (torch.max(label) + 1).long().item()
        if normalize != -1:
            graph.ndata['feat'] = F.normalize(graph.ndata['feat'], dim=1, p=normalize)
        if undirected:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        if self_loop:
            graph = graph.remove_self_loop().add_self_loop()
    else:
        raise NotImplementedError
    return graph, label, class_num


def load_fixed_data_split(dataname, split_idx):
    """load fixed split for benchmark dataset, train/val/test is 48%/32%/20%.
    Parameters
    ----------
    dataname: str
        dataset name.
    split_idx: int
        id of split plan.
    """
    splits_file_path = f'./data/splits/{dataname}_splits.pt'
    splits_file = torch.load(splits_file_path)
    train_mask_list = splits_file['train']
    val_mask_list = splits_file['val']
    test_mask_list = splits_file['test']
    
    train_mask = torch.BoolTensor(train_mask_list[split_idx])
    val_mask = torch.BoolTensor(val_mask_list[split_idx])
    test_mask = torch.BoolTensor(test_mask_list[split_idx])
    return train_mask, val_mask, test_mask

