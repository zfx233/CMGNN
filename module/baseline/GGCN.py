# Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks, ICDM 2022
# The source of model's main code: https://github.com/Yujun-Yan/Heterophily_and_oversmoothing

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
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor



class GGCN(nn.Module):

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
        self.use_decay = args.use_decay
        if self.use_decay:
            self.decay = args.decay_rate
            self.exponent = args.exponent
        self.degree_precompute = None
        self.use_degree = args.use_degree
        self.use_sparse = args.use_sparse
        self.use_norm = args.use_bn or args.use_ln
        self.device = device
        self.epochs = args.epochs
        self.patience = args.patience
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        if self.use_norm:
            self.norms = nn.ModuleList()
        if args.use_bn:
            for _ in range(args.nlayers-1):
                self.norms.append(nn.BatchNorm1d(nhidden))
        if args.use_ln:
            for _ in range(args.nlayers-1):
                self.norms.append(nn.LayerNorm(nhidden))

        #---------------- Layer -------------------
        self.convs = nn.ModuleList()
        if self.use_sparse:
            model_sel = GGCNlayer_SP
        else:
            model_sel = GGCNlayer
        self.convs.append(model_sel(in_features, args.nhidden, args.use_degree, args.use_sign, args.use_decay, args.scale_init, args.deg_intercept_init))
        for _ in range(args.nlayers-2):
            self.convs.append(model_sel(args.nhidden, args.nhidden, args.use_degree, args.use_sign, args.use_decay, args.scale_init, args.deg_intercept_init))
        self.convs.append(model_sel(args.nhidden, class_num, args.use_degree, args.use_sign, args.use_decay, args.scale_init, args.deg_intercept_init))
        self.fcn = nn.Linear(in_features, args.nhidden)
        self.act_fn = F.elu
        

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
        if self.use_sparse:
            adj_norm = sys_normalized_adjacency(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)
        else:
            adj = torch.tensor(adj.todense(), device=self.device, dtype=torch.float)
            adj = F.normalize(adj, dim=1, p=1)

        
        best_epoch = 0
        best_acc = 0.
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
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
        if self.use_degree:
            if self.degree_precompute is None:
                if self.use_sparse:
                    self.precompute_degree_s(adj)
                else:
                    self.precompute_degree_d(adj)
        x = F.dropout(x, self.dropout, training=self.training)
        layer_previous = self.fcn(x)
        layer_previous = self.act_fn(layer_previous)
        layer_inner = self.convs[0](x, adj, self.degree_precompute)

        Z = None
        for i,con in enumerate(self.convs[1:]):
            if self.use_norm:
                layer_inner = self.norms[i](layer_inner)
            layer_inner = self.act_fn(layer_inner)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if i==0:
                layer_previous = layer_inner + layer_previous
            else:
                if self.use_decay:
                    coeff = math.log(self.decay/(i+2)**self.exponent+1)
                else:
                    coeff = 1
                layer_previous = coeff*layer_inner + layer_previous
            layer_inner = con(layer_previous, adj, self.degree_precompute)
        if return_Z:
            return layer_inner, F.log_softmax(layer_inner, dim=1)
        return F.log_softmax(layer_inner, dim=1)


    def test(self, X, adj, labels, index_list):
        self.eval()
        with torch.no_grad():
            C = self.forward(X, adj)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list


    def predict(self, graph):
        self.eval()
        graph = graph.to(self.device)
        X = graph.ndata['feat']
        adj = graph.adj(scipy_fmt='csr')
        if self.use_sparse:
            adj_norm = sys_normalized_adjacency(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)
        else:
            adj = torch.tensor(adj.todense(), device=self.device, dtype=torch.float)
            adj = F.normalize(adj, dim=1, p=1)
        with torch.no_grad():
            Z, C = self.forward(X, adj, return_Z=True)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()

    def precompute_degree_d(self, adj):
        diag_adj = torch.diag(adj)
        diag_adj = torch.unsqueeze(diag_adj, dim=1)
        self.degree_precompute = diag_adj/torch.max(adj, 1e-9*torch.ones_like(adj))-1
    
    def precompute_degree_s(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_diag_ind = (adj_i[0,:]==adj_i[1,:])
        adj_diag = adj_v[adj_diag_ind]
        v_new = torch.zeros_like(adj_v)
        for i in range(adj_i.shape[1]):
            v_new[i] = adj_diag[adj_i[0,i]]/adj_v[i]-1
        self.degree_precompute = torch.sparse.FloatTensor(adj_i, v_new, adj.size())



class GGCNlayer_SP(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer_SP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5,0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init,0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            self.adj_remove_diag = None
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init*torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)
    
    def precompute_adj_wo_diag(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_wo_diag_ind = (adj_i[0,:]!=adj_i[1,:])
        self.adj_remove_diag = torch.sparse.FloatTensor(adj_i[:,adj_wo_diag_ind], adj_v[adj_wo_diag_ind], adj.size())
                        
    def non_linear_degree(self, a, b, s):
        i = s._indices()
        v = s._values()
        return torch.sparse.FloatTensor(i, self.sftpls(a*v+b), s.size())
    
    def get_sparse_att(self, adj, Wh):
        i = adj._indices()
        Wh_1 = Wh[i[0,:],:]
        Wh_2 = Wh[i[1,:],:]
        sim_vec = F.cosine_similarity(Wh_1, Wh_2)
        sim_vec_pos = F.relu(sim_vec)
        sim_vec_neg = -F.relu(-sim_vec)
        return torch.sparse.FloatTensor(i, sim_vec_pos, adj.size()), torch.sparse.FloatTensor(i, sim_vec_neg, adj.size())
    
    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.non_linear_degree(self.deg_coeff[0], self.deg_coeff[1], degree_precompute)

        Wh = self.fcn(h)
        if self.use_sign:
            if self.adj_remove_diag is None:
                self.precompute_adj_wo_diag(adj)
        if self.use_sign:
            e_pos, e_neg = self.get_sparse_att(adj, Wh)
            if self.use_degree:
                attention_pos = self.adj_remove_diag*sc*e_pos
                attention_neg = self.adj_remove_diag*sc*e_neg
            else:
                attention_pos = self.adj_remove_diag*e_pos
                attention_neg = self.adj_remove_diag*e_neg
            
            prop_pos = torch.sparse.mm(attention_pos, Wh)
            prop_neg = torch.sparse.mm(attention_neg, Wh)
        
            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)

        else:
            if self.use_degree:
                prop = torch.sparse.mm(adj*sc, Wh)
            else:
                prop = torch.sparse.mm(adj, Wh)
            
            result = prop
        return result

class GGCNlayer(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5,0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init,0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init*torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)


    
    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.deg_coeff[0]*degree_precompute+self.deg_coeff[1]
            sc = self.sftpls(sc)

        Wh = self.fcn(h)
        if self.use_sign:
            prod = torch.matmul(Wh, torch.transpose(Wh, 0, 1))
            sq = torch.unsqueeze(torch.diag(prod),1)
            scaling = torch.matmul(sq, torch.transpose(sq, 0, 1))
            e = prod/torch.max(torch.sqrt(scaling),1e-9*torch.ones_like(scaling))
            e = e-torch.diag(torch.diag(e))
            if self.use_degree:
                attention = e*adj*sc
            else:
                attention = e*adj
            
            attention_pos = F.relu(attention)
            attention_neg = -F.relu(-attention)
            prop_pos = torch.matmul(attention_pos, Wh)
            prop_neg = torch.matmul(attention_neg, Wh)
        
            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)

        else:
            if self.use_degree:
                prop = torch.matmul(adj*sc, Wh)
            else:
                prop = torch.matmul(adj, Wh)
            
            result = prop
                 
        return result