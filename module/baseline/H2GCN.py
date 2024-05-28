# Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs, NuralIPS 2020
# The source of model's main code: https://github.com/GemsLab/H2GCN

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
from sklearn.metrics import accuracy_score as ACC
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor


class H2GCN(nn.Module):

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
        self.epochs = args.epochs
        self.device = device
        self.patience = args.patience
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.k = args.k
        self.use_relu = args.use_relu

        #---------------- Model -------------------
        self.act = F.relu if args.use_relu else lambda x: x
        self.w_embed = nn.Parameter(
            torch.zeros(size=(in_features, args.hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * args.hidden_dim, class_num)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

        
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

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
            loss = F.nll_loss(output[self.train_mask], labels[self.train_mask])
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

        
    def forward(self, x: FloatTensor, adj: torch.sparse.Tensor, return_Z=False) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)

        if return_Z:
            return r_final, torch.softmax(torch.mm(r_final, self.w_classify), dim=1)
        return torch.softmax(torch.mm(r_final, self.w_classify), dim=1)
        

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

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)