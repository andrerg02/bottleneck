import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.utils import remove_self_loops
import numpy as np

from common import GNN_TYPE

class GraphModel(torch.nn.Module):
    def __init__(self, args, gnn_type, num_layers, dim0, h_dim, out_dim, last_layer_fully_adjacent,
                 unroll, layer_norm, use_activation, use_residual):
        super(GraphModel, self).__init__()
        self.gnn_type = getattr(GNN_TYPE, gnn_type) if type(gnn_type) is str else gnn_type
        self.unroll = unroll
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = self.device

        self.num_layers = num_layers
        self.layer0_keys = nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim*args.d)
        self.layer0_values = nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim*args.d)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        #self.lin1 = nn.Linear(h_dim, 32 * 2)
        #self.lin2 = nn.Linear(32 * 2, h_dim)

        if unroll:
            self.layers.append(self.gnn_type.get_layer(args,
                in_dim=h_dim,
                out_dim=h_dim))
        else:
            for i in range(num_layers):
                self.layers.append(self.gnn_type.get_layer(args,
                    in_dim=h_dim,
                    out_dim=h_dim))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim*args.d))

        self.out_dim = out_dim
        # self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim, bias=False)
        self.out_layer = nn.Linear(in_features=h_dim*args.d, out_features=out_dim + 1, bias=False)

    def forward(self, data, reff=False):
        x, edge_index, batch, roots = data.x, data.edge_index, data.batch, data.root_mask

        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed


        reff_per_layer = torch.zeros((self.num_layers,), device=self.device)
        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            new_x = x
            if self.last_layer_fully_adjacent and i == self.num_layers - 1:
                root_indices = torch.nonzero(roots, as_tuple=False).squeeze(-1)
                target_roots = root_indices.index_select(dim=0, index=batch)
                source_nodes = torch.arange(0, data.num_nodes).to(self.device)
                edges = torch.stack([source_nodes, target_roots], dim=0)

            else:
                edges = edge_index

            #full_edges = torch.cat([edges, edges.flip(0)], dim=1).unique(dim=1)
            #edges = full_edges
            edges = remove_self_loops(edges)[0]

            if 'Discrete' in layer.__class__.__name__ and 'Flat' not in layer.__class__.__name__:
                layer.get_edge_dependend_stuff(edges, new_x)
            # else:
            #     layer.compute_maps_idx(edges)
            #layer.to(new_x.device)

            new_x, reff_values = layer(new_x, edges, data, reff=reff)

            reff_sum, mean_reff, std_reff = reff_values
            reff_per_layer[i] = reff_sum
                
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        root_nodes = x[roots]
        logits = self.out_layer(root_nodes)
        # logits = F.linear(root_nodes, self.layer0_values.weight)
        if reff:
            return logits, reff_per_layer
        return logits
