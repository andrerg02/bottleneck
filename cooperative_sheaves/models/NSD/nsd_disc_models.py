# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse

from torch import nn
from cooperative_sheaves.models.NSD.sheaf_base import SheafDiffusion
import cooperative_sheaves.models.NSD.laplacian_builders as lb
from cooperative_sheaves.models.NSD.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant, LocalConcatFlatSheafLearnerVariant

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_add

import cooperative_sheaves.models.NSD.laplace as lap
from cooperative_sheaves.models.orthogonal import Orthogonal

import numpy as np

from scipy.sparse import lil_matrix
import random

import numpy as np
import scipy.sparse as sp
from scipy.linalg import pinv

class DiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class DiscreteBundleSheafDiffusion(SheafDiffusion, MessagePassing):

    def __init__(self, args):
        #super(DiscreteBundleSheafDiffusion, self).__init__(args)
        SheafDiffusion.__init__(self, args)
        MessagePassing.__init__(self, aggr='add', flow='target_to_source', node_dim=0)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        for i in range(self.layers):
            if self.right_weights:
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
            else:
                self.lin_right_weights.append(nn.Identity())
            if self.left_weights:
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
            else:
                self.lin_left_weights.append(nn.Identity())

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        #self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.orth_transform = Orthogonal(self.d, self.orth_trans)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    
    def get_edge_dependend_stuff(self, edge_index, x):
        self.graph_size = x.size(0)
        undirected_edges = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)

        if self.use_edge_weights:
            for _ in range(self.layers):
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, undirected_edges))
            self.weight_learners.to(self.device)

        # self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
        #     self.graph_size, undirected_edges, d=self.d, add_hp=self.add_hp,
        #     add_lp=self.add_lp, orth_map=self.orth_trans)
        
        self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(undirected_edges)
        self.left_idx, self.right_idx = self.left_right_idx
        self.tril_row, self.tril_col = self.vertex_tril_idx
        self.new_edge_index = torch.cat([self.vertex_tril_idx, self.vertex_tril_idx.flip(0)], dim=1)

        full_left_right_idx, _ = lap.compute_left_right_map_index(undirected_edges, full_matrix=True)
        _, self.full_right_index = full_left_right_idx
        
        self.deg = degree(edge_index[0])

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def align_edges_with_dummy(self, edge_index, num_edges=None):
        row, col = edge_index
        if num_edges is None:
            num_edges = edge_index.size(1) // 2
        N = num_edges
        dummy = self.graph_size 

        a = torch.minimum(row, col)
        b = torch.maximum(row, col)
        key = a * (N + 1) + b

        uniq, inv = torch.unique(key, return_inverse=True)
        K = uniq.numel()

        row_gt = torch.full((K,), dummy, device=row.device, dtype=torch.long)
        col_gt = torch.full((K,), dummy, device=row.device, dtype=torch.long)
        row_lt = torch.full((K,), dummy, device=row.device, dtype=torch.long)
        col_lt = torch.full((K,), dummy, device=row.device, dtype=torch.long)

        m_gt = row > col
        m_lt = row < col

        idx_gt = inv[m_gt]
        row_gt[idx_gt] = row[m_gt]
        col_gt[idx_gt] = col[m_gt]

        idx_lt = inv[m_lt]
        row_lt[idx_lt] = row[m_lt]
        col_lt[idx_lt] = col[m_lt]

        edge_index_gt_aligned = torch.stack([row_gt, col_gt], dim=0)
        edge_index_lt_aligned = torch.stack([row_lt, col_lt], dim=0)

        return edge_index_gt_aligned, edge_index_lt_aligned, dummy
    
    def restriction_maps_builder(self, maps, edge_index, edge_weights):
        row, _ = edge_index
        edge_weights = edge_weights.squeeze(-1) if edge_weights is not None else None
        maps = self.orth_transform(maps)

        if edge_weights is not None:
            diag_maps = scatter_add(edge_weights ** 2, row, dim=0, dim_size=self.graph_size)
            maps = maps * edge_weights[:, None, None]
        else:
            diag_maps = degree(row, num_nodes=self.graph_size)

        left_maps = maps[self.left_idx]
        right_maps = maps[self.right_idx]

        diag_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm = diag_sqrt_inv[self.tril_row]
        right_norm = diag_sqrt_inv[self.tril_col]

        norm_left_maps = left_norm.view(-1, 1, 1) * left_maps
        norm_right_maps = right_norm.view(-1, 1, 1) * right_maps

        maps_prod = -torch.bmm(norm_left_maps.transpose(-2,-1), norm_right_maps)
        
        norm_D = diag_maps * diag_sqrt_inv**2 

        return norm_D, maps_prod
    
    def align_edges_and_maps(self, edge_index, maps, dummy):
        dummies = torch.ones_like(edge_index) * dummy
        mask = edge_index != dummies

        edge_index = torch.cat([edge_index[0][mask[0]][None, :],
                                edge_index[1][mask[1]][None, :]], dim=0)

        maps = maps[mask[0]]

        return edge_index, maps


    def total_sheaf_effective_resistance(self, L_G_pinv, R, F_maps):
        n = L_G_pinv.shape[0]
        d = F_maps[0].shape[0] 
        ones_d = np.ones(d)
        s_vectors = [F_map.T @ ones_d for F_map in F_maps]
        
        M = np.zeros((n, n))
        for u in range(n):
            for v in range(n):
                M[u, v] = s_vectors[u].T @ s_vectors[v]
        
        frobenius_term = np.trace(M @ L_G_pinv)
        
        R_F = d * R - frobenius_term
        
        return R_F, R, frobenius_term

    def batched_effective_resistance(self, data, maps):
        from torch_geometric.utils import to_dense_adj
        results = []

        graphs = data.to_data_list()
        offset = 0
        for g in graphs:
            num_nodes = g.num_nodes
            g_maps = maps[offset:offset+num_nodes].detach().numpy()
            offset += num_nodes

            L_G_pinv = g.L_G_pinv
            R = g.R
            reff, _, _ = self.total_sheaf_effective_resistance(L_G_pinv, R, g_maps)

            results.append(reff)

        return np.sum(results)
    
    def forward(self, x, edge_index, data, reff=False):
        torch.set_printoptions(linewidth=200)
        #x = x.to(torch.float64)
        self.edge_index = edge_index
        undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)

        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        upper_edge_index, lower_edge_index, dummy = self.align_edges_with_dummy(edge_index)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, undirected_edge_index)
                edge_weights = self.weight_learners[layer](x_maps, undirected_edge_index) if self.use_edge_weights else None
                #L, trans_maps = self.laplacian_builder(maps, edge_weights)
                #self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            D, maps_prod = self.restriction_maps_builder(maps, undirected_edge_index, edge_weights)

            y = x.clone()
            y = y.reshape(self.graph_size, self.d, self.hidden_channels)
            deg = degree(undirected_edge_index[0], num_nodes=self.graph_size)
            Dx = D[:, None, None] * y * deg.pow(-1)[:, None, None]

            if edge_index.size(1) != undirected_edge_index.size(1):
                u_edge_index, u_maps_prod = self.align_edges_and_maps(upper_edge_index, maps_prod, dummy)
                l_edge_index, l_maps_prod = self.align_edges_and_maps(lower_edge_index, maps_prod, dummy)

                y1 = self.propagate(u_edge_index, x=y, diag=Dx, Ft=u_maps_prod.transpose(-2,-1))
                y2 = self.propagate(l_edge_index, x=y, diag=Dx, Ft=l_maps_prod)
                y = y1 + y2
            else:
                y1 = self.propagate(upper_edge_index, x=y, diag=Dx, Ft=maps_prod.transpose(-2,-1))
                y2 = self.propagate(lower_edge_index, x=y, diag=Dx, Ft=maps_prod)
                y = y1 + y2

            y = y.view(self.graph_size * self.final_d, -1)

            # Use the adjacency matrix rather than the diagonal
            # x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            # print(torch.allclose(x,y, atol=1e-6))
            # print(torch.norm(x-y, p=2))

            x = y

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        sum_reff, mean_reff, var_reff = 0, 0, 0
        if reff:
            sum_reff = self.batched_effective_resistance(data, Dx)
            # print(f"Effective Resistance: {sum_reff}")

        x = x.reshape(self.graph_size, -1)
        #x = self.lin2(x)
        return x, (sum_reff, mean_reff, var_reff)#F.log_softmax(x, dim=1)

    def message(self, x_j, diag_i, Ft):
        msg = Ft @ x_j

        return diag_i + msg

class DiscreteFlatBundleSheafDiffusion(SheafDiffusion, MessagePassing):

    def __init__(self, args):
        #super(DiscreteBundleSheafDiffusion, self).__init__(args)
        SheafDiffusion.__init__(self, args)
        MessagePassing.__init__(self, aggr='add', flow='source_to_target', node_dim=0)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatFlatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.orth_transform = Orthogonal(self.d, self.orth_trans)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    
    def get_edge_dependend_stuff(self, edge_index, x):
        undirected_edges = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)
        self.graph_size = x.size(0)
        if self.use_edge_weights:
            self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, undirected_edges))
            self.weight_learners.to(self.device)
        self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(undirected_edges)
        self.left_idx, self.right_idx = self.left_right_idx
        self.tril_row, self.tril_col = self.vertex_tril_idx
        self.new_edge_index = torch.cat([self.vertex_tril_idx, self.vertex_tril_idx.flip(0)], dim=1)
        
        full_left_right_idx, _ = lap.compute_left_right_map_index(undirected_edges, full_matrix=True)
        _, self.full_right_index = full_left_right_idx

        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, undirected_edges, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)
        self.laplacian_builder.to(self.device)
        
        self.deg = degree(edge_index[0])
    
    def restriction_maps_builder(self, F, edge_index, edge_weights):#, edge_weights):
        row, col = edge_index
        undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)
        un_row, un_col = undirected_edge_index
        edge_weights = edge_weights.squeeze(-1) if edge_weights is not None else None

        maps = self.orth_transform(F)

        if edge_weights is not None:
            diag_maps = scatter_add(edge_weights ** 2, un_row, dim=0, dim_size=self.graph_size)
        else:
            diag_maps = degree(row, num_nodes=self.graph_size)
            
        # left_maps = maps[row]
        # right_maps = maps[col]

        diag_sqrt_inv = (diag_maps + 1).pow(-0.5)

        # norm_left_maps = diag_sqrt_inv[row].view(-1, 1, 1) * left_maps
        # norm_right_maps = diag_sqrt_inv[col].view(-1, 1, 1) * right_maps

        norm_maps = diag_sqrt_inv.view(-1, 1, 1) * maps

        #final_left_maps = norm_left_maps * edge_weights[row][:, None, None]
        #final_right_maps = norm_right_maps * edge_weights[col][:, None, None]

        #maps_prod = -torch.bmm(norm_left_maps.transpose(-2,-1), norm_right_maps)
        #print(f"This is their normalized product:\n {maps_prod}")

        norm_D = diag_maps * diag_sqrt_inv**2 

        return norm_D, norm_maps#, final_left_maps, final_right_maps
    
    def total_sheaf_effective_resistance(self, L_G_pinv, R, F_maps):
        n = L_G_pinv.shape[0]
        d = F_maps[0].shape[0] 
        ones_d = np.ones(d)
        s_vectors = [F_map.T @ ones_d for F_map in F_maps]
        
        M = np.zeros((n, n))
        for u in range(n):
            for v in range(n):
                M[u, v] = s_vectors[u].T @ s_vectors[v]
        
        frobenius_term = np.trace(M @ L_G_pinv)
        
        R_F = d * R - frobenius_term
        
        return R_F, R, frobenius_term

    def batched_effective_resistance(self, data, maps):
        from torch_geometric.utils import to_dense_adj
        results = []

        graphs = data.to_data_list()
        offset = 0
        for g in graphs:
            num_nodes = g.num_nodes
            g_maps = maps[offset:offset+num_nodes].detach().numpy()
            offset += num_nodes

            L_G_pinv = g.L_G_pinv
            R = g.R
            reff, _, _ = self.total_sheaf_effective_resistance(L_G_pinv, R, g_maps)

            results.append(reff)

        return np.sum(results)


    def forward(self, x, edge_index, data, reff=False):
        #x = x.to(torch.float64)
        self.edge_index = edge_index
        self.undirected_edges = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        full_left_right_idx, _,  = lap.compute_left_right_map_index(self.undirected_edges, full_matrix=True)
        _, full_right_index = full_left_right_idx
        #print(f"This is full_right_index: {full_right_index}")

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                #print(self.undirected_edges)
                maps = self.sheaf_learners[layer](x_maps, self.undirected_edges)
                edge_weights = self.weight_learners[layer](x_maps, self.undirected_edges, full_right_index) if self.use_edge_weights else None
                #L, trans_maps, maps, self.left_idx, self.right_idx, self.vertex_tril_idx = self.laplacian_builder(maps, edge_weights)
                #self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            
            D, maps = self.restriction_maps_builder(maps, edge_index, edge_weights)
            
            #print(trans_maps.view(prod.size(0), self.d, self.d).shape, prod.shape)

            #print(trans_maps.view(prod.size(0), self.d, self.d))
            #print(prod)
            # aaaaaa = []
            # for i in range(D.size(0)):
            #     aaaaaa.append(D[i] * torch.eye(self.d, device=self.device))

            # aaaaaa = torch.block_diag(*aaaaaa)
            
            #F11 = F_row[0].transpose(-2, -1) @ F_col[0]

            #dense_L = torch_sparse.to_torch_sparse(L[0], L[1], self.graph_size*self.d, self.graph_size*self.d).to_dense()
            #print(f"Dense Laplacian: \n {dense_L}")
            #print(f"Prod of rest maps: \n {maps_prod}")
            #print(D)
            #print(f"Diag matrix: \n {aaaaaa}")
            #print(torch.equal(dense_L, aaaaaa))
            #print(torch.norm(dense_L - aaaaaa, p=2))

            #maps_prod = torch.cat([prod, prod.transpose(-2,-1)],dim=0)
            y = x.clone()
            y = y.reshape(self.graph_size, self.d, self.hidden_channels)
            deg = degree(self.undirected_edges[0], num_nodes=self.graph_size)
            Dx = D[:, None, None] * y * deg.pow(-1)[:, None, None]
            #print(edge_index)
            Fy = maps @ y
            y = self.propagate(edge_index, x=Fy, diag=Dx, Ft=maps.transpose(-2,-1))
                               #D=D, F_row=F_row, F_col=F_col)

            # Use the adjacency matrix rather than the diagonal
            #x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            
            #z = dense_L @ z
            #z = z.view(self.graph_size * self.final_d, -1)

            x = y.view(self.graph_size * self.final_d, -1)
            #print(x)
            #print(y)
            ##print(z)
            #print(torch.allclose(x,y, atol=1e-6))
            #print(torch.norm(x-y, p=2))
            

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0


        #x = self.lin2(x)
        sum_reff, mean_reff, var_reff = 0, 0, 0
        if reff:
            sum_reff = self.batched_effective_resistance(data, Dx)
            # print(f"Effective Resistance: {sum_reff}")

        x = x.reshape(self.graph_size, -1)
        #x = self.lin2(x)
        return x, (sum_reff, mean_reff, var_reff)#F.log_softmax(x, dim=1)

    def message(self, x_j, diag_i, Ft_i):
    #def message(self, x_i, x_j, D_i, F_row, F_col):
        #diag = D_i[:, None, None] * x_i
        msg = Ft_i @ x_j
        #msg = F_row.transpose(-2,-1) @ F_col @ x_j

        return diag_i - msg

class DiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)