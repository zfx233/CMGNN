# MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing, ICML 2019
# The source of model's main code: https://github.com/benedekrozemberczki/MixHop-and-N-GCN

import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
import copy
import time
from torch import FloatTensor
import torch_sparse
from torch_sparse import spmm
from sklearn.metrics import accuracy_score as ACC


class MixHop(nn.Module):

    def __init__(
            self,
            in_features: int,
            class_num: int,
            device,
            args,
        ) -> None:
        super().__init__()
        #------------- Parameters ----------------
        self.feature_num = in_features
        self.class_num = class_num
        self.device = device
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience

        self.layers_1 = args.layers_1
        self.layers_2 = args.layers_2
        self.dropout = args.dropout
        self.lambd = args.lambd
        self.cut_off = args.cut_off
        self.budget = args.budget

        #---------------- Model -------------------


    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape
        adj = graph.adj(scipy_fmt='coo')

        feature = transform_feature(X, self.device)
        propagation_matrix = transform_adj(adj, self.device)

        self.fit_(feature, propagation_matrix, labels, base_run=True)
        # self.evaluate_architecture()
        # self.reset_architecture()
        # self.fit_(feature, propagation_matrix, labels, base_run=False)


    def fit_(self, feature, propagation_matrix, labels, base_run=True):
        self.calculate_layer_sizes()
        self.setup_layer_structure(base_run)
        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        best_epoch = 0
        best_acc = 0.
        cnt = 0
        best_state_dict = None

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(feature, propagation_matrix)
            loss = F.nll_loss(output[self.train_mask], labels[self.train_mask])
            if base_run == True:
                loss = loss + self.calculate_group_loss()
            else:
                loss = loss + self.calculate_loss()
            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(feature, propagation_matrix, labels, [self.train_mask, self.valid_mask, self.test_mask])

            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                print(f'\nEpoch:{epoch}, Loss:{loss.item()}')
                print(f'train acc: {train_acc:.3f} valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}')

            else:
                cnt += 1
                if cnt == self.patience:
                    print(f"Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc}")
                    break
        self.load_state_dict(best_state_dict)
        self.best_epoch = best_epoch
        
    def forward(self, features, normalized_adjacency_matrix, return_Z=False):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features_1 = torch.cat([self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_2 = torch.cat([self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)], dim=1)
        predictions = torch.nn.functional.log_softmax(self.fully_connected(abstract_features_2), dim=1)
        if return_Z:
            return abstract_features_2, predictions
        return predictions
        

    def test(self, feature, propagation_matrix, labels, index_list):
        self.eval()
        with torch.no_grad():
            C = self.forward(feature, propagation_matrix)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list


    def predict(self, graph):
        self.eval()
        graph = graph.to(self.device)
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape
        adj = graph.adj(scipy_fmt='coo')
        feature = transform_feature(X, self.device)
        propagation_matrix = transform_adj(adj, self.device)

        with torch.no_grad():
            Z, C = self.forward(feature, propagation_matrix, return_Z=True)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.layers_1)
        self.abstract_feature_number_2 = sum(self.layers_2)
        self.order_1 = len(self.layers_1)
        self.order_2 = len(self.layers_2)

    def setup_layer_structure(self, base_run):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        if not base_run:
            del self.upper_layers
            del self.bottom_layers

        self.upper_layers = [SparseNGCNLayer(self.feature_num, self.layers_1[i-1], i, self.dropout) for i in range(1, self.order_1+1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1, self.layers_2[i-1], i, self.dropout) for i in range(1, self.order_2+1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_num)

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.lambd*loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.lambd*loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.lambd*loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.lambd*loss_bottom
        return weight_loss

    def evaluate_architecture(self):
        """
        Making a choice about the optimal layer sizes.
        """
        print("The best architecture is:\n")
        self.layer_sizes = dict()

        self.layer_sizes["upper"] = []

        for layer in self.upper_layers:
            norms = torch.norm(layer.weight_matrix**2, dim=0)
            norms = norms[norms < self.cut_off]
            self.layer_sizes["upper"].append(norms.shape[0])

        self.layer_sizes["bottom"] = []

        for layer in self.bottom_layers:
            norms = torch.norm(layer.weight_matrix**2, dim=0)
            norms = norms[norms < self.cut_off]
            self.layer_sizes["bottom"].append(norms.shape[0])

        self.layer_sizes["upper"] = [int(self.budget*layer_size/sum(self.layer_sizes["upper"]))  for layer_size in self.layer_sizes["upper"]]
        self.layer_sizes["bottom"] = [int(self.budget*layer_size/sum(self.layer_sizes["bottom"]))  for layer_size in self.layer_sizes["bottom"]]
        print("Layer 1.: "+str(tuple(self.layer_sizes["upper"])))
        print("Layer 2.: "+str(tuple(self.layer_sizes["bottom"])))

    def reset_architecture(self):
        """
        Changing the layer sizes.
        """
        print("\nResetting the architecture.\n")
        self.layers_1 = self.layer_sizes["upper"]
        self.layers_2 = self.layer_sizes["bottom"]


def transform_feature(X, device):
    """
    :return out_features: Dict with index and value tensor.
    """
    features = sp.coo_matrix(X.cpu().numpy())
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features


def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def transform_adj(A, device):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator


class SparseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        feature_count, _ = torch.max(features["indices"],dim=1)
        feature_count = feature_count + 1
        base_features = spmm(features["indices"], features["values"], feature_count[0],
                             feature_count[1], self.weight_matrix)

        base_features = base_features + self.bias

        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)

        base_features = torch.nn.functional.relu(base_features)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        return base_features

class DenseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Dense Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.mm(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        return base_features

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
