# Powerful Graph Convolutioal Networks with Adaptive Propagation Mechanism for Homophily and Heterophily, AAAI 2022
# The source of model's main code: https://github.com/hedongxiao-tju/HOG-GCN

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
import math
import copy
import time
from sklearn.metrics import accuracy_score as ACC
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor


class HOGGCN(nn.Module):

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

		#---------------- Model -------------------
		self.GCN1 = GCN_homo(in_features, args.adj, args.nhid1, args.nhid2, args.dropout, device)
		self.a = nn.Parameter(torch.zeros(size=(args.nhid2, 1)))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)
		self.attention = Attention(args.nhid2)
		self.tanh = nn.Tanh()
		self.MLP = nn.Sequential(
			#nn.Linear(nhid2, nhid2),
			nn.Linear(args.nhid2, class_num),
			nn.LogSoftmax(dim=1)
		)

		self.model_MLP = MLP_hog(
			n_feat=in_features,
			n_hid=args.nhid2,
			nclass=class_num,
			dropout=self.dropout
		)


	def fit(self, graph, labels, train_mask, val_mask, test_mask):
		graph = graph.to(self.device)
		labels = labels.to(self.device)
		self.train_mask = train_mask.to(self.device)
		self.valid_mask = val_mask.to(self.device)
		self.test_mask = test_mask.to(self.device)
		self.to(self.device)
		X = graph.ndata["feat"]
		n_nodes, _ = X.shape
		graph = graph.remove_self_loop().add_self_loop()
		adj = graph.adj(scipy_fmt='csr')
		adj = torch.tensor(adj.todense(), device=self.device, dtype=torch.float32)


		optimizer_mlp = optim.Adam(self.model_MLP.parameters(), lr=self.lr, weight_decay=0.02)
		mlp_acc_val_best = 0

		## MLP pre-train
		for i in range(10):
			self.model_MLP.train()
			optimizer_mlp.zero_grad()
			output = self.model_MLP(X)
			loss = F.nll_loss(output[train_mask], labels[train_mask])
			acc = accuracy(output[train_mask], labels[train_mask])
			loss.backward()
			optimizer_mlp.step()
			acc_val = accuracy(output[val_mask], labels[val_mask])
			acc_test = accuracy(output[test_mask], labels[test_mask])
			print('epoch:{}'.format(i+1),
				'loss: {:.4f}'.format(loss.item()),
				'acc: {:.4f}'.format(acc.item()),
				'val: {:.4f}'.format(acc_val.item()),
				'test: {:.4f}'.format(acc_test.item()))

		self.si_adj = adj
		self.bi_adj = adj.mm(adj)
		self.labels_for_lp = one_hot_embedding(labels, labels.max().item() + 1, output).type(torch.FloatTensor).to(self.device)


		optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
		best_epoch = 0
		best_acc = 0.
		cnt = 0
		best_state_dict = None

		for epoch in range(self.epochs):
			self.train()
			self.model_MLP.train()
			optimizer.zero_grad()
			optimizer_mlp.zero_grad()

			output_mlp = self.model_MLP(X)
			output, y_hat, adj_mask, emb = self.forward(X, output_mlp)

			loss_mlp = F.nll_loss(output_mlp[self.train_mask], labels[self.train_mask])
			loss_lp = F.nll_loss(y_hat[self.train_mask], labels[self.train_mask])
			loss = loss_mlp + F.nll_loss(output[self.train_mask], labels[self.train_mask]) + 1*loss_lp

			loss.backward()
			optimizer.step()
			optimizer_mlp.step()

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

		
	def forward(self, x, output):
		emb, y_hat, mask = self.GCN1(x, self.si_adj, self.bi_adj, output, self.labels_for_lp)
		output = self.MLP(emb)
		return output, F.log_softmax(y_hat, dim=1), mask, emb
		

	def test(self, X, adj, labels, index_list):
		self.eval()
		self.model_MLP.eval()
		with torch.no_grad():
			output_mlp = self.model_MLP(X)
			C, _, _, _ = self.forward(X, output_mlp)
			y_pred = torch.argmax(C, dim=1)
		acc_list = []
		for index in index_list:
			acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
		return acc_list


	def predict(self, graph):
		self.eval()
		self.model_MLP.eval()
		graph = graph.to(self.device)
		X = graph.ndata['feat']

		with torch.no_grad():
			output_mlp = self.model_MLP(X)
			C, _, _, Z = self.forward(X, output_mlp)
			y_pred = torch.argmax(C, dim=1)

		return y_pred.cpu(), C.cpu(), Z.cpu()



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def one_hot_embedding(labels, num_classes, soft):
	"""Embedding labels to one-hot form.

	Args:
		labels: (LongTensor) class labels, sized [N,].
		num_classes: (int) number of classes.

	Returns:
		(tensor) encoded labels, sized [N, #classes].
	"""
	soft = torch.argmax(soft.exp(), dim=1)
	y = torch.eye(num_classes)
	return y[soft]



class GCN_homo(nn.Module):
	def __init__(self, nfeat, adj, nhid, out, dropout, device):
		super(GCN_homo, self).__init__()
		self.gc1 = GraphConvolution_homo(nfeat, adj, nhid, device)
		self.gc2 = GraphConvolution_homo(nhid, adj, nhid, device)
		self.gc3 = GraphConvolution_homo(nhid, adj, out, device)
		self.dropout = dropout


	def forward(self, x, adj, bi_adj, output, labels_for_lp):
		x, y_hat, mask = self.gc1(x, adj, bi_adj, output, labels_for_lp)
		x = F.relu(x)
		x = F.dropout(x, self.dropout, training = self.training)
		#x_2 = F.relu(self.gc2(x, adj, bi_adj, output))
		#x_2 = F.dropout(x_2, self.dropout, training=self.training)
		x_3, y_hat, mask = self.gc3(x, adj, bi_adj, output, labels_for_lp)
		#return torch.cat((x, x_2), dim=1)
		return x_3, y_hat, mask


class Attention(nn.Module):
	def __init__(self, in_size, hidden_size=32):
		super(Attention, self).__init__()

		self.project = nn.Sequential(
			nn.Linear(in_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, 1, bias=False)
		)

	def forward(self, z):
		w = self.project(z)
		beta = torch.softmax(w, dim=1)
		return (beta * z).sum(1), beta


class MLP_hog(nn.Module):
	def __init__(self, n_feat, n_hid, nclass, dropout):
		super(MLP_hog, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(n_feat, n_hid),
			#nn.ReLU(),
			nn.Linear(n_hid, nclass),
			nn.LogSoftmax(dim=1)
		)

	def forward(self, x):
		return self.mlp(x)

	def get_emb(self, x):
		return self.mlp[0](x).detach()


class GraphConvolution_homo(nn.Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, adj, out_features, device, bias=True):
		super(GraphConvolution_homo, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		self.weight_bi = Parameter(torch.FloatTensor(in_features, out_features))
		self.w = Parameter(torch.FloatTensor(1))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.adjacency_mask = Parameter(adj.clone())
		self.reset_parameters()
		self.device = device


	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		stdv_bi = 1. / math.sqrt(self.weight_bi.size(1))
		self.weight_bi.data.uniform_(-stdv_bi, stdv_bi)
		self.w.data.uniform_(0.5, 1)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj, bi_adj, output, labels_for_lp):

		new_bi = bi_adj.clone()
		new_bi = new_bi * self.adjacency_mask
		new_bi = F.normalize(new_bi, p=1, dim=1)
		identity = torch.eye(adj.shape[0]).to(self.device)
		output = output.exp()
		homo_matrix = torch.matmul(output, output.t())
		homo_matrix = 0.4 * homo_matrix + 1 * new_bi
		y_hat = torch.mm(new_bi, labels_for_lp)

		bi_adj = torch.mul(bi_adj, homo_matrix)


		with torch.no_grad():
			bi_row_sum = torch.sum(bi_adj, dim=1, keepdim=True)
			bi_r_inv = torch.pow(bi_row_sum, -1).flatten()  # np.power(rowsum, -1).flatten()
			bi_r_inv[torch.isinf(bi_r_inv)] = 0.
			bi_r_mat_inv = torch.diag(bi_r_inv)
		bi_adj = torch.matmul(bi_r_mat_inv, bi_adj)


		support = torch.mm(input, self.weight)
		support_bi = torch.mm(input, self.weight_bi)
		output = torch.spmm(identity, support)
		output_bi = torch.spmm(bi_adj, support_bi)
		output = output + torch.mul(self.w, output_bi)

		if self.bias is not None:
			return output + self.bias, y_hat, homo_matrix
		else:
			return output, y_hat, homo_matrix

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
				+ str(self.in_features) + ' -> ' \
				+ str(self.out_features) + ')'

