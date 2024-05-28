# Finding Global Homophily in Graph Neural Networks When Meeting Heterophily, ICML 2022
# The source of model's main code: https://github.com/RecklessRonan/GloGNN

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
from sklearn.metrics import accuracy_score as ACC
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor


class GloGNN(nn.Module):

    def __init__(
            self,
            in_features: int,
            class_num: int,
            device,
            args,
        ) -> None:
        super().__init__()
        #------------- Parameters ----------------
        self.fc1 = nn.Linear(in_features, args.nhid)
        self.fc2 = nn.Linear(args.nhid, class_num)
        self.fc3 = nn.Linear(args.nnodes, args.nhid)
        self.nclass = class_num
        self.dropout = args.dropout
        self.alpha = torch.tensor(args.alpha).to(device)
        self.beta = torch.tensor(args.beta).to(device)
        self.gamma = torch.tensor(args.gamma).to(device)
        self.delta = torch.tensor(args.delta).to(device)
        self.norm_layers = args.norm_layers
        self.orders = args.orders
        self.device = device
        self.class_eye = torch.eye(class_num).to(device)
        self.nodes_eye = torch.eye(args.nnodes).to(device)
        self.epochs = args.epochs
        self.patience = args.early_stopping
        self.lr = args.lr
        self.l2_coef = args.l2_coef


        #---------------- Model -------------------
        self.orders_weight = Parameter(
            torch.ones(self.orders, 1) / self.orders, requires_grad=True
        )
        # use kaiming_normal to initialize the weight matrix in Orders3
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(self.nclass, self.orders), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(self.orders, self.orders), requires_grad=True
        )
        # use diag matirx to initialize the second norm layer
        self.diag_weight = Parameter(
            torch.ones(self.nclass, 1) / self.nclass, requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()

        if args.norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if args.orders_func_id == 1:
            self.order_func = self.order_func1
        elif args.orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3


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

        
    def forward(self, x, adj, return_Z=False):
        xX = F.dropout(x, self.dropout, training=self.training)
        xX = self.fc1(x)
        xA = self.fc3(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        for _ in range(self.norm_layers):
            # adj_drop = F.dropout(adj, self.dropout, training=self.training)
            x = self.norm(x, h0, adj)
        
        if return_Z:
            return x, F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)
        

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


    def norm_func1(self, x, h0, adj):
        # print('norm_func1 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        # print('norm_func2 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0

        # calculate z
        xx = torch.mm(x, x.t())
        hx = torch.mm(h0, x.t())
        # print('adj', adj.shape)
        # print('orders_weight', self.orders_weight[0].shape)
        adj = adj.to_dense()
        adjk = adj
        a_sum = adjk * self.orders_weight[0]
        for i in range(1, self.orders):
            adjk = torch.mm(adjk, adj)
            a_sum += adjk * self.orders_weight[i]
        z = torch.mm(coe1 * xx + self.beta * a_sum - self.gamma * coe1 * hx,
                     torch.inverse(coe1 * coe1 * xx + (self.alpha + self.beta) * self.nodes_eye))
        # print(z.shape)
        # print(z)
        return res

    def order_func1(self, x, res, adj):
        # Orders1
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # Orders2
        tmp_orders = torch.spmm(adj, res)
        # print('tmp_orders', tmp_orders.shape)
        # print('orders_weight', self.orders_weight[0].shape)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        # Orders3
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        # orders_para = torch.mm(x, self.orders_weight_matrix)
        orders_para = torch.transpose(orders_para, 0, 1)
        tmp_orders = torch.spmm(adj, res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders