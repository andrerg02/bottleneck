from enum import Enum, auto

from tasks.dictionary_lookup import DictionaryLookupDataset

from torch import nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv
from cooperative_sheaves.models.coopshv_model import CoopSheafDiffusion, NSDConv, CSNNConv
from cooperative_sheaves.models.NSD.nsd_disc_models import (
    DiscreteBundleSheafDiffusion,
    DiscreteFlatBundleSheafDiffusion,
    DiscreteFlatGeneralSheafDiffusion)



class Task(Enum):
    NEIGHBORS_MATCH = auto()

    @staticmethod
    def from_string(s):
        try:
            return Task[s]
        except KeyError:
            raise ValueError()

    def get_dataset(self, depth, train_fraction):
        if self is Task.NEIGHBORS_MATCH:
            dataset = DictionaryLookupDataset(depth)
        else:
            dataset = None

        return dataset.generate_data(train_fraction)


class GNN_TYPE(Enum):
    GCN = auto()
    GGNN = auto()
    GIN = auto()
    GAT = auto()
    CSN = auto()
    ONSD = auto()
    FNSD = auto()
    FGNSD = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, args, in_dim, out_dim):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim)
        elif self is GNN_TYPE.GGNN:
            return GatedGraphConv(out_channels=out_dim, num_layers=1)
        elif self is GNN_TYPE.GIN:
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # 4-heads, although the paper by Velickovic et al. had used 6-8 heads.
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            num_heads = 4
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        elif self is GNN_TYPE.CSN:
            args.input_dim = in_dim
            args.output_dim = out_dim
            return CoopSheafDiffusion(args)
        elif self is GNN_TYPE.ONSD:
            args.input_dim = in_dim
            args.output_dim = out_dim
            return DiscreteBundleSheafDiffusion(args)
        elif self is GNN_TYPE.FNSD:
            args.input_dim = in_dim
            args.output_dim = out_dim
            return DiscreteFlatBundleSheafDiffusion(args)
        elif self is GNN_TYPE.FGNSD:
            args.input_dim = in_dim
            args.output_dim = out_dim
            return DiscreteFlatGeneralSheafDiffusion(args)


class STOP(Enum):
    TRAIN = auto()
    TEST = auto()

    @staticmethod
    def from_string(s):
        try:
            return STOP[s]
        except KeyError:
            raise ValueError()


def one_hot(key, depth):
    return [1 if i == key else 0 for i in range(depth)]
