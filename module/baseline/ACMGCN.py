# Revisiting Heterophily for Graph Neural Networks, NuralIPS 2022
# The source of model's main code: https://github.com/SitaoLuan/ACM-GNN

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
from torch.nn.parameter import Parameter
import random
import copy
from sklearn.metrics import accuracy_score as ACC
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor



class ACMGCN(nn.Module):

    def __init__(
            self,
            in_features: int,
            class_num: int,
            device,
            args,
        ) -> None:
        super().__init__()
        #------------- Parameters ----------------
        if args.acm_model == "acmgcnpp":
            self.mlpX = MLP_acm(in_features, args.nhid, args.nhid, num_layers=args.init_layers_X, dropout=0)
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.acm_model, self.structure_info, self.nlayers, self.nnodes = (
            args.acm_model,
            args.structure_info,
            args.nlayers,
            args.nnodes,
        )
        self.dropout = args.dropout
        self.device = device
        self.epochs = args.epochs
        self.patience = args.patience
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        #---------------- Layer -------------------
        if (
            self.acm_model == "acmgcn"
            or self.acm_model == "acmgcnp"
            or self.acm_model == "acmgcnpp"
        ):
            self.gcns.append(
                GraphConvolution(
                    in_features,
                    args.nhid,
                    args.nnodes,
                    acm_model=args.acm_model,
                    variant=args.variant,
                    structure_info=args.structure_info,
                )
            )
            self.gcns.append(
                GraphConvolution(
                    1 * args.nhid,
                    class_num,
                    args.nnodes,
                    acm_model=args.acm_model,
                    output_layer=1,
                    variant=args.variant,
                    structure_info=args.structure_info,
                )
            )
        elif self.acm_model == "acmsgc":
            self.gcns.append(GraphConvolution(in_features, class_num, acm_model=args.acm_model))
        elif self.acm_model == "acmsnowball":
            for k in range(nlayers):
                self.gcns.append(
                    GraphConvolution(
                        k * args.nhid + in_features, args.nhid, acm_model=args.acm_model, variant=args.variant
                    )
                )
            self.gcns.append(
                GraphConvolution(
                    args.nlayers * args.nhid + in_features,
                    class_num,
                    acm_model=args.acm_model,
                    variant=args.variant,
                )
            )

        self.fea_param, self.xX_param = Parameter(
            torch.FloatTensor(1, 1).to(device)
        ), Parameter(torch.FloatTensor(1, 1).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        if self.acm_model == "acmgcnpp":
            self.mlpX.reset_parameters()
        else:
            pass

    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)
        self.X = graph.ndata["feat"]
        nnodes, _ = self.X.shape
        adj = graph.adj(scipy_fmt='csr')
        adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj)

        if (self.acm_model == "acmgcnp" or self.acm_model == "acmgcnpp") and (
            self.structure_info == 1
        ):
            pass
        else:
            self.X = normalize_tensor(self.X)
        
        if self.structure_info:
            adj_low = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense())
            self.adj_high = (torch.eye(nnodes) - adj_low).to(self.device).to_sparse()
            self.adj_low = adj_low.to(self.device)
            self.adj_low_unnormalized = adj_low_unnormalized.to(self.device)
        else:
            adj_low = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense())
            self.adj_high = (torch.eye(nnodes) - adj_low).to(self.device).to_sparse()
            self.adj_low = adj_low.to(self.device)
            self.adj_low_unnormalized = None

        best_epoch = 0
        best_acc = 0.
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        best_state_dict = None

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.X,)
            loss = F.nll_loss(output[train_mask], labels[train_mask].to(self.device))
            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(self.X, labels, [self.train_mask, self.valid_mask, self.test_mask])

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

    def forward(self, x, return_Z=False):
        if (
            self.acm_model == "acmgcn"
            or self.acm_model == "acmsgc"
            or self.acm_model == "acmsnowball"
            or self.acm_model == "acmgcnp"
            or self.acm_model == "acmgcnpp"
        ):
            x = F.dropout(x, self.dropout, training=self.training)
            if self.acm_model == "acmgcnpp":
                xX = F.dropout(
                    F.relu(self.mlpX(x, input_tensor=True)),
                    self.dropout,
                    training=self.training,
                )
        if self.acm_model == "acmsnowball":
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(
                        F.dropout(
                            F.relu(layer(x, adj_low, adj_high)),
                            self.dropout,
                            training=self.training,
                        )
                    )
                else:
                    list_output_blocks.append(
                        F.dropout(
                            F.relu(
                                layer(
                                    torch.cat([x] + list_output_blocks[0:layer_num], 1),
                                    adj_low,
                                    adj_high,
                                )
                            ),
                            self.dropout,
                            training=self.training,
                        )
                    )
            return self.gcns[-1](
                torch.cat([x] + list_output_blocks, 1), adj_low, adj_high
            )

        fea1 = self.gcns[0](x, self.adj_low, self.adj_high, self.adj_low_unnormalized)

        if (
            self.acm_model == "acmgcn"
            or self.acm_model == "acmgcnp"
            or self.acm_model == "acmgcnpp"
        ):
            fea1 = F.dropout((F.relu(fea1)), self.dropout, training=self.training)

            if self.acm_model == "acmgcnpp":
                fea2 = self.gcns[1](fea1 + xX, self.adj_low, self.adj_high, self.adj_low_unnormalized)
            else:
                fea2 = self.gcns[1](fea1, self.adj_low, self.adj_high, self.adj_low_unnormalized)
        
        if return_Z:
            return fea2, F.log_softmax(fea2, dim=1)

        return F.log_softmax(fea2, dim=1)


    def test(self, X, labels, index_list):
        self.eval()
        with torch.no_grad():
            C = self.forward(self.X)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list


    def predict(self, graph):
        self.eval()
        with torch.no_grad():
            Z, C = self.forward(self.X, return_Z=True)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()




class GraphConvolution(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        nnodes,
        acm_model,
        output_layer=0,
        variant=False,
        structure_info=0,
    ):
        super(GraphConvolution, self).__init__()
        (
            self.in_features,
            self.out_features,
            self.output_layer,
            self.acm_model,
            self.structure_info,
            self.variant,
        ) = (
            in_features,
            out_features,
            output_layer,
            acm_model,
            structure_info,
            variant,
        )
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(torch.FloatTensor(in_features, out_features)),
            Parameter(torch.FloatTensor(in_features, out_features)),
            Parameter(torch.FloatTensor(in_features, out_features)),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(torch.FloatTensor(1 * out_features, 1)),
            Parameter(torch.FloatTensor(1 * out_features, 1)),
            Parameter(torch.FloatTensor(1 * out_features, 1)),
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.FloatTensor(1 * out_features, 1)
        )
        self.struc_low = Parameter(torch.FloatTensor(nnodes, out_features))
        if self.structure_info == 0:
            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        else:
            self.att_vec = Parameter(torch.FloatTensor(4, 4))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()

    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        if self.acm_model == "acmgcn+" or self.acm_model == "acmgcn++":
            output_low, output_high, output_mlp = (
                self.layer_norm_low(output_low),
                self.layer_norm_high(output_high),
                self.layer_norm_mlp(output_mlp),
            )
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def attention4(self, output_low, output_high, output_mlp, struc_low):
        T = 4
        if self.acm_model == "acmgcn+" or self.acm_model == "acmgcn++":
            feature_concat = torch.cat(
                [
                    torch.mm(self.layer_norm_low(output_low), self.att_vec_low),
                    torch.mm(self.layer_norm_high(output_high), self.att_vec_high),
                    torch.mm(self.layer_norm_mlp(output_mlp), self.att_vec_mlp),
                    torch.mm(self.layer_norm_struc_low(struc_low), self.att_struc_low),
                ],
                1,
            )
        else:
            feature_concat = torch.cat(
                [
                    torch.mm((output_low), self.att_vec_low),
                    torch.mm((output_high), self.att_vec_high),
                    torch.mm((output_mlp), self.att_vec_mlp),
                    torch.mm((struc_low), self.att_struc_low),
                ],
                1,
            )

        logits = torch.mm(torch.sigmoid(feature_concat), self.att_vec) / T

        att = torch.softmax(logits, 1)
        return (
            att[:, 0][:, None],
            att[:, 1][:, None],
            att[:, 2][:, None],
            att[:, 3][:, None],
        )

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        output = 0
        if self.acm_model == "mlp":
            output_mlp = torch.mm(input, self.weight_mlp)
            return output_mlp
        elif self.acm_model == "sgc" or self.acm_model == "gcn":
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.acm_model == "acmsgc":
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            output_high = torch.spmm(adj_high, torch.mm(input, self.weight_high))
            output_mlp = torch.mm(input, self.weight_mlp)

            self.att_low, self.att_high, self.att_mlp = self.attention3(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
            )
        else:
            if self.variant:

                output_low = torch.spmm(
                    adj_low, F.relu(torch.mm(input, self.weight_low))
                )

                output_high = torch.spmm(
                    adj_high, F.relu(torch.mm(input, self.weight_high))
                )
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            else:
                output_low = F.relu(
                    torch.spmm(adj_low, (torch.mm(input, self.weight_low)))
                )
                output_high = F.relu(
                    torch.spmm(adj_high, (torch.mm(input, self.weight_high)))
                )
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            if self.acm_model == "acmgcn" or self.acm_model == "acmsnowball":
                self.att_low, self.att_high, self.att_mlp = self.attention3(
                    (output_low), (output_high), (output_mlp)
                )
                return 3 * (
                    self.att_low * output_low
                    + self.att_high * output_high
                    + self.att_mlp * output_mlp
                )
            else:
                if self.structure_info:
                    output_struc_low = F.relu(
                        torch.mm(adj_low_unnormalized, self.struc_low)
                    )
                    (
                        self.att_low,
                        self.att_high,
                        self.att_mlp,
                        self.att_struc_vec_low,
                    ) = self.attention4(
                        (output_low), (output_high), (output_mlp), output_struc_low
                    )
                    return 1 * (
                        self.att_low * output_low
                        + self.att_high * output_high
                        + self.att_mlp * output_mlp
                        + self.att_struc_vec_low * output_struc_low
                    )
                else:
                    self.att_low, self.att_high, self.att_mlp = self.attention3(
                        (output_low), (output_high), (output_mlp)
                    )
                    return 3 * (
                        self.att_low * output_low
                        + self.att_high * output_high
                        + self.att_mlp * output_mlp
                    )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MLP_acm(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5
    ):
        super(MLP_acm, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
            self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph["node_feat"]
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x



def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx