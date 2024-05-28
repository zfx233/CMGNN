# Simple and Deep Graph Convolutional Networks, ICML 2020
# The source of model's main code: https://github.com/Yujun-Yan/Heterophily_and_oversmoothing/blob/main/model.py

import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import math
import torch.optim as optim
import copy
import time
from sklearn.metrics import accuracy_score as ACC
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor



class GCNII(nn.Module):

    def __init__(
            self,
            in_features: int,
            class_num: int,
            device,
            args,
        ) -> None:
        super().__init__()
        #------------- Parameters ----------------
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.lammbda = args.lammbda
        self.device = device
        self.epochs = args.epochs
        self.patience = args.patience
        self.lr = args.lr
        self.l2_coef = args.l2_coef

        #---------------- Model -------------------
        self.convs = nn.ModuleList()
        for _ in range(args.layers):
            self.convs.append(GraphConvolution(args.nhidden, args.nhidden, variant=args.variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_features, args.nhidden))
        self.fcs.append(nn.Linear(args.nhidden, class_num))
        self.act_fn = nn.ReLU()


    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape
        adj = graph.adj(scipy_fmt='csr')
        adj_norm = sys_normalized_adjacency(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        best_epoch = 0
        best_acc = 0.
        cnt = 0
        best_state_dict = None

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(X, adj)
            loss = F.nll_loss(output[train_mask], labels[train_mask].to(self.device))
            loss.backward()
            optimizer.step()
        
            [train_acc, valid_acc, test_acc] = self.test(X, adj, labels, [self.train_mask, self.valid_mask, self.test_mask])
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

    
    def forward(self, x, adj, return_Z=False):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj, _layers[0], self.lammbda, self.alpha, i+1))
        Z = layer_inner.clone()
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        if return_Z:
            return Z, F.log_softmax(layer_inner, dim=1)
        return F.log_softmax(layer_inner, dim=1)

    def test(self, X, adj, labels, index_list):
        self.eval()
        with torch.no_grad():
            Z = self.forward(X, adj)
            y_pred = torch.argmax(Z, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph):
        self.eval()
        graph = graph.to(self.device)
        X = graph.ndata['feat']
        adj = graph.adj(scipy_fmt='csr')
        adj_norm = sys_normalized_adjacency(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)


        with torch.no_grad():
            Z, C = self.forward(X, adj, return_Z=True)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()



class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lammbda, alpha, l):
        theta = math.log(lammbda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output