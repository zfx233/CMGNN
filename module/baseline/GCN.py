# Semi-supervised Classification with Graph Convolutional Networks.

import os
import dgl
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from dgl.nn import GraphConv
import math
import time
import random
import copy
from sklearn.metrics import accuracy_score as ACC



class GCN(nn.Module):

    def __init__(
            self,
            in_features: int,
            class_num: int,
            device,
            args,
        ) -> None:
        super().__init__()
        #------------- Parameters ----------------
        self.class_num = class_num
        self.device = device
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience
        self.n_layers = args.layers
        self.dropout= args.dropout

        #---------------- Layer -------------------
        layers = []
        pre_dim = in_features
        for i in range(self.n_layers):
            if i == self.n_layers-1:
                now_dim = self.class_num
            else:
                now_dim = args.nhidden
            layers.append(GraphConv(pre_dim, now_dim))
            pre_dim = now_dim

        self.model = nn.ModuleList(layers)
        # for module in self.modules():
        #     init_weights(module)


    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)
        
        graph = graph.remove_self_loop().add_self_loop()
        adj = graph.adj(scipy_fmt='csr')
        adj = torch.tensor(adj.todense(), device=self.device, dtype=torch.float)
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape

        best_epoch = 0
        best_acc = 0.
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_state_dict = None

        for epoch in range(self.epochs):
            self.train()

            Z = self.forward(graph, X)
            loss = loss_fn(Z[self.train_mask], labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(graph, X, labels, [self.train_mask, self.valid_mask, self.test_mask])

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


    def forward(self, graph, X, return_Z=False):
        Z = X
        Z_ = None
        for i in range(self.n_layers):
            if i == self.n_layers-1:
                Z_ = Z.clone()
            Z = F.dropout(Z, self.dropout, training=self.training)
            Z = self.model[i](graph, Z)
        C = F.softmax(Z, dim=1)
        if return_Z:
            return Z_, C
        return C

    def test(self, graph, X, labels, index_list):
        self.eval()
        with torch.no_grad():
            Z = self.forward(graph, X)
            y_pred = torch.argmax(Z, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph):
        self.eval()
        graph = graph.remove_self_loop().add_self_loop()
        graph = graph.to(self.device)
        X = graph.ndata['feat']
        with torch.no_grad():
            Z, C = self.forward(graph, X, return_Z=True)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()
