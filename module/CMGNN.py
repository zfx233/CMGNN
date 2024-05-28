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
from utils import row_normalized_adjacency, sparse_mx_to_torch_sparse_tensor



class CCPGNN(nn.Module):

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
        self.device = device
        self.epochs = args.epochs
        self.patience = args.patience
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.class_num = class_num
        self.structure_info = args.structure_info
        self.lambda_ = args.lambda_

        #---------------- Model -------------------

        self.ccp_layers = nn.ModuleList()
        for _ in range(args.layers):
            self.ccp_layers.append(CCPLayer(args.nhidden, args.nhidden, class_num, args.variant))
        self.fc_layers = nn.ModuleList()
        if self.structure_info:
            self.backbone_x = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(in_features, args.nhidden)
            )
            self.backbone_a = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(args.nnodes + 1, args.nhidden),
            )
            self.fc_layers.append(nn.Linear(args.nhidden * 2, args.nhidden))
        else:
            self.fc_layers.append(nn.Linear(in_features, args.nhidden))
        self.fc_layers.append(nn.Linear(args.nhidden * (args.layers + 1), class_num))
        self.act_fn = nn.ReLU()


    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)
        graph = graph.remove_self_loop()
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape
        adj = graph.adj(scipy_fmt='csr')
        adj_norm, deg = row_normalized_adjacency(adj, return_deg=True)
        deg = torch.tensor(deg, dtype=torch.float).to(self.device)
        adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)

        deg = torch.cat([deg, torch.zeros((self.class_num, 1)).to(self.device)], dim=0)
        self.add_supplement_neighbors(X, adj, deg, labels)
        labels = torch.cat([labels, torch.arange(self.class_num, device=self.device)], dim=0)
        # print("M:", self.M)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_epoch = 0
        best_acc = 0.
        cnt = 0
        best_state_dict = None

        t_start = time.time()
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            Z, C = self.forward(self.X, self.deg)
            loss_d = self.discrimination_loss(Z, self.M)
            loss = loss_d + loss_fn(C[self.train_mask], labels[self.train_mask].to(self.device))
            loss.backward()
            optimizer.step()
        
            [train_acc, valid_acc, test_acc] = self.test(self.X, labels, [self.train_mask, self.valid_mask, self.test_mask])
            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                print(f'\nEpoch:{epoch}, Loss:{loss.item()}')
                print(f'train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}')
                C[self.train_mask] = self.C_real[self.train_mask]
                self.C = nn.Parameter(C, requires_grad=False)
                self.update_connect_preference()
            else:
                cnt += 1
                if cnt == self.patience:
                    print(f"Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc}")
                    break
        t_finish = time.time()
        print("\n10 epoch cost: {:.4f}s\n".format((t_finish - t_start)/(epoch+1)*10))
        self.load_state_dict(best_state_dict)
        self.best_epoch = best_epoch
        print("M:", self.M)
    
    def add_supplement_neighbors(self, X, adj, deg, labels):
        n = X.shape[0]
        self.adj_o = torch.sparse_coo_tensor(adj._indices(), adj._values(), size=(n+self.class_num, n+self.class_num)).to(self.device)
        self.adj_s = torch.ones((n + self.class_num, self.class_num)).to(self.device)

        C_real = F.one_hot(labels, num_classes=self.class_num).float().to(self.device)
        C_sup = torch.eye(self.class_num).to(self.device)
        
        self.C_real = torch.cat([C_real, C_sup], dim=0)
        self.train_mask = torch.cat([self.train_mask, torch.BoolTensor([True]*self.class_num).to(self.device)], dim=0)
        self.valid_mask = torch.cat([self.valid_mask, torch.BoolTensor([False]*self.class_num).to(self.device)], dim=0)
        self.test_mask = torch.cat([self.test_mask, torch.BoolTensor([False]*self.class_num).to(self.device)], dim=0)

        X_sup = self.get_suppletment_nodes(X)
        self.X = torch.cat([X, X_sup], dim=0)

        self.deg_weight(deg.reshape(-1))
        self.deg = deg
        C = torch.ones((n+self.class_num, self.class_num)).to(self.device) / self.class_num

        C[self.train_mask] = self.C_real[self.train_mask]
        self.C = nn.Parameter(C, requires_grad=False)
        self.update_connect_preference()

    def get_suppletment_nodes(self, X):
        X_sup = torch.zeros((self.class_num, X.shape[1]), device=self.device)
        index = torch.nonzero(self.train_mask).reshape(-1)
        labels = torch.argmax(self.C_real, dim=1)
        available_labels = labels[index]
        for i in range(self.class_num):
            id_i = torch.nonzero(available_labels == i).reshape(-1)
            X_sup[i, :] = torch.mean(X[id_i], dim=0)
        return X_sup

    def deg_weight(self, deg):
        weight = torch.zeros_like(deg, dtype=torch.float, device=self.device)
        index = (deg<=self.class_num)
        weight[index] = deg[index] / self.class_num / 2
        index = (deg<=self.class_num * 3)
        weight[index] = deg[index] / self.class_num / 4 + 0.25
        index = (deg > self.class_num * 3)
        weight[index] = 1 
        self.w_d = weight
    
    def calc_entropy(self, p):
        logp = torch.log(p + 1e-8)
        H = - torch.sum(p * logp, dim=1)
        return math.log(self.class_num) - H
    
    def update_connect_preference(self):
        g = self.calc_entropy(self.C)
        self.g = Parameter(g, requires_grad=False)
        gC = self.g.view(-1, 1) * self.C
        C_nb = F.normalize(torch.spmm(self.adj_o, gC), dim=1, p=1)
        C_t_weight = F.normalize((self.C * self.w_d.view(-1, 1) * self.g.view(-1, 1)).t(), dim=1, p=1)
        M = F.normalize(torch.matmul(C_t_weight, C_nb), dim=1, p=1)
        self.M = nn.Parameter(M, requires_grad=False)

    def discrimination_loss(self, Z, M):
        n = Z.shape[0] - self.class_num
        MZ = F.normalize(torch.matmul(M, Z[n:, :]), dim=1, p=2)
        sim = torch.matmul(MZ, MZ.t())
        loss = (torch.sum(sim) - torch.sum(torch.diag(sim))) / self.class_num / (self.class_num - 1)
        return loss * self.lambda_

    def forward(self, X, deg):
        if self.structure_info:
            n = X.shape[0] - self.class_num
            X = F.dropout(X, self.dropout, training=self.training)
            A = self.adj_o.to_dense()[:, :n]
            A = F.dropout(A, self.dropout, training=self.training)
            H_x = self.backbone_x(X)
            H_a = self.backbone_a(torch.cat([A, deg], dim=1))
            X = torch.cat([H_x, H_a], dim=1)
            
        H_list = []
        X = F.dropout(X, self.dropout, training=self.training)
        H = self.act_fn(self.fc_layers[0](X))
        H_list.append(H)
        for i, layer in enumerate(self.ccp_layers):
            # H = F.dropout(H, self.dropout, training=self.training)
            H = self.act_fn(layer(H, self.adj_o, self.adj_s, self.C, self.M, deg))
            H_list.append(H)
        Z = torch.cat(H_list, dim=1)
        Z = F.dropout(Z, self.dropout, training=self.training)
        C = self.fc_layers[-1](Z)
        C = F.softmax(C, dim=1)
        return Z, C

    def test(self, X, labels, index_list):
        self.eval()
        with torch.no_grad():
            Z, C = self.forward(X, self.deg)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph):
        self.eval()
        graph = graph.to(self.device)
        graph = graph.remove_self_loop()
        X = graph.ndata['feat']
        adj = graph.adj(scipy_fmt='csr')

        with torch.no_grad():
            Z, C = self.forward(self.X, self.deg)
            C = C[:X.shape[0]]
            y_pred = torch.argmax(C, dim=1)

        # return y_pred.cpu(), C.cpu(), Z[:X.shape[0]].cpu()
        return y_pred.cpu(), self.M.cpu(), Z[:X.shape[0]].cpu()



class CCPLayer(nn.Module):

    def __init__(self, in_features, out_features, class_num, variant):
        super(CCPLayer, self).__init__() 
        self.class_num = class_num
        self.variant = variant
        self.w_0 = Parameter(torch.FloatTensor(in_features, out_features))
        self.w_1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.w_2 = Parameter(torch.FloatTensor(in_features, out_features))
        self.alpha_learner = nn.Sequential(
            nn.Linear(out_features * 3 + 1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 3),
            nn.Softmax(dim=1),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w_0.size(1))
        self.w_0.data.uniform_(-stdv, stdv)
        self.w_1.data.uniform_(-stdv, stdv)
        self.w_2.data.uniform_(-stdv, stdv)

    def forward(self, H, adj, adj_s, C, M, deg):
        n = H.shape[0] - self.class_num

        B_s_ = torch.matmul(C, M)
        AB = F.normalize(adj_s * B_s_, dim=1, p=1)

        Z_0 = F.relu(torch.matmul(H, self.w_0))
        if self.variant:
            Z_1 = torch.spmm(adj, F.relu(torch.matmul(H, self.w_1)))
            Z_2 = torch.spmm(AB, F.relu(torch.matmul(H[n:, :], self.w_2)))
        else:
            Z_1 = F.relu(torch.spmm(adj, torch.matmul(H, self.w_1)))
            Z_2 = F.relu(torch.spmm(AB, torch.matmul(H[n:, :], self.w_2)))

        alpha = self.alpha_learner(torch.cat([Z_0, Z_1, Z_2, deg], dim=1))
        Z = alpha[:, 0].view(-1, 1) * Z_0 + alpha[:, 1].view(-1, 1) * Z_1 + alpha[:, 2].view(-1, 1) * Z_2

        return Z
