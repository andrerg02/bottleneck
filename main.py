from argparse import ArgumentParser
from attrdict import AttrDict

from experiment import Experiment
from common import Task, GNN_TYPE, STOP
from distutils.util import strtobool

import wandb

def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", dest="task", default=Task.NEIGHBORS_MATCH, type=Task.from_string, choices=list(Task),
                        required=False)
    parser.add_argument("--type", dest="type", default=GNN_TYPE.GCN, type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                        required=False)
    parser.add_argument("--dim", dest="dim", default=32, type=int, required=False)
    parser.add_argument("--depth", dest="depth", default=3, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=None, type=int, required=False)
    parser.add_argument("--train_fraction", dest="train_fraction", default=0.8, type=float, required=False)
    parser.add_argument("--max_epochs", dest="max_epochs", default=50000, type=int, required=False)
    parser.add_argument("--eval_every", dest="eval_every", default=100, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=1024, type=int, required=False)
    parser.add_argument("--accum_grad", dest="accum_grad", default=1, type=int, required=False)
    parser.add_argument("--stop", dest="stop", default=STOP.TRAIN, type=STOP.from_string, choices=list(STOP),
                        required=False)
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)
    parser.add_argument("--loader_workers", dest="loader_workers", default=0, type=int, required=False)
    parser.add_argument('--last_layer_fully_adjacent', action='store_true')
    parser.add_argument('--no_layer_norm', action='store_true')
    parser.add_argument('--no_activation', action='store_true')
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--unroll', action='store_true', help='use the same weights across GNN layers')

    # Flat models stuff

    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--layers', type=int, default=1,)
    parser.add_argument('--num_heads', type=int, default=1,)
    parser.add_argument("--gnn_layers", type=int, default=0,)
    parser.add_argument("--gnn_hidden", type=int, default=32,)
    parser.add_argument("--pe_size", type=int, default=0,)
    parser.add_argument('--linear_emb', type=str2bool, default=False,)
    parser.add_argument('--gnn_type', type=str, default='SAGE',)
    parser.add_argument('--gnn_default', type=int, default=1,)
    parser.add_argument('--gnn_residual', type=str2bool, default=False,)
    parser.add_argument('--layer_norm', type=str2bool, default=False,)
    parser.add_argument('--batch_norm', type=str2bool, default=False,)
    parser.add_argument('--conformal', type=str2bool, default=False,)



    # NSD stuff
    
    parser.add_argument('--normalised', dest='normalised', type=str2bool, default=True)
    parser.add_argument('--deg_normalised', dest='deg_normalised', type=str2bool, default=False)
    parser.add_argument('--linear', dest='linear', type=str2bool, default=False,
                        help="Whether to learn a new Laplacian at each step.")
    parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--second_linear', dest='second_linear', type=str2bool, default=False)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=False)
    parser.add_argument('--use_bias', dest='use_bias', type=str2bool, default=True)

    args = parser.parse_args()
    args.num_layers = args.depth + 1# if args.num_layers is None else args.num_layers
    args.gnn_layers = args.depth + 1
    wandb.init(project="FlatNSD_bottleneck", config=args)
    Experiment(wandb.config).run()


def get_fake_args(
        task=Task.NEIGHBORS_MATCH,
        type=GNN_TYPE.GCN,
        dim=32,
        depth=3,
        num_layers=None,
        train_fraction=0.8,
        max_epochs=50000,
        eval_every=100,
        batch_size=1024,
        accum_grad=1,
        patience=20,
        stop=STOP.TRAIN,
        loader_workers=0,
        last_layer_fully_adjacent=False,
        no_layer_norm=False,
        no_activation=False,
        no_residual=False,
        unroll=False,
):
    return AttrDict({
        'task': task,
        'type': type,
        'dim': dim,
        'depth': depth,
        'num_layers': num_layers,
        'train_fraction': train_fraction,
        'max_epochs': max_epochs,
        'eval_every': eval_every,
        'batch_size': batch_size,
        'accum_grad': accum_grad,
        'stop': stop,
        'patience': patience,
        'loader_workers': loader_workers,
        'last_layer_fully_adjacent': last_layer_fully_adjacent,
        'no_layer_norm': no_layer_norm,
        'no_activation': no_activation,
        'no_residual': no_residual,
        'unroll': unroll,
        'd': 2,
        'layers': 1,
        'gnn_layers': num_layers,
        'gnn_hidden': dim,
        'hidden_channels': dim,
        'left_weights': True,
        'right_weights': True,
        'orth': 'householder',
        'pe_size': 0,
        'sheaf_act': 'tanh',
        'input_dropout': 0.0,
        'dropout': 0.0,
        'use_act': False,
        'num_heads': 1,
        'linear_emb': False,
        'gnn_type': 'SAGE',
        'gnn_default': 1,
        'gnn_residual': False,
        'layer_norm': False,
        'batch_norm': False,
        'conformal': True,
        #NSD Specific stuff
        'add_lp': False,
        'add_hp': False,
        'normalised': True,
        'deg_normalised': False,
        'linear': False,
        'sparse_learner': True,
        'second_linear': False,
        'edge_weights': False,
    })
