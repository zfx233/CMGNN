import os
import dgl
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import time
import random
import copy
from sklearn.metrics import accuracy_score as ACC



class MLP(nn.Module):

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
        self.dropout = args.dropout

        #---------------- Layer -------------------
        self.MLP = nn.Sequential(
            nn.Linear(in_features, args.nhidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(args.nhidden, self.class_num),
            nn.Softmax(dim=1)
        )

    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape
        adj = graph.adj(scipy_fmt='csr')
        adj = torch.tensor(adj.todense(), device=self.device, dtype=torch.float)
        
        best_epoch = 0
        best_acc = 0.
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_state_dict = None

        for epoch in range(self.epochs):
            self.train()
            C = self.forward(X)
            loss = loss_fn(C[self.train_mask], labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(X, labels, [self.train_mask, self.valid_mask, self.test_mask])

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


    def forward(self, X, return_Z=False):
        X = F.dropout(X, self.dropout, training=self.training)
        Z = self.MLP(X)
        Z = F.dropout(Z, self.dropout, training=self.training)
        C = self.classifier(Z)
        if return_Z: 
            return Z, C
        return C

    def test(self, X, labels, index_list):
        self.eval()
        with torch.no_grad():
            C = self.forward(X)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list


    def predict(self, graph):
        self.eval()
        graph = graph.to(self.device)
        X = graph.ndata['feat']
        with torch.no_grad():
            Z, C = self.forward(X, return_Z=True)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()
