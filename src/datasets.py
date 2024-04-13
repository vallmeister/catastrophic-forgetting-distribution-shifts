from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch_geometric
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset, EllipticBitcoinTemporalDataset


def get_dblp():
    path = Path('./data/dblp-hard/')

    x = np.load(f'{path}/x.npy')
    y = np.load(f'{path}/y.npy')
    node_year = np.load(f'{path}/t.npy')

    nx_graph = nx.read_adjlist(f'{path}/adjlist.txt', nodetype=int)
    data = torch_geometric.utils.from_networkx(nx_graph)
    data.x = torch.tensor(x)
    data.y = torch.unsqueeze(torch.tensor(y), 1)
    data.node_year = torch.unsqueeze(torch.tensor(node_year), 1)

    return data


def get_elliptic():
    # Labels {0: licit, 1: illicit, 2: unlabeled}
    elliptic = EllipticBitcoinDataset(root='./data/Elliptic/')
    return elliptic


def get_elliptic_temporal(t):
    elliptic = EllipticBitcoinTemporalDataset(root='./data/EllipticTemporal/', t=t)
    return elliptic


"""
Split Elliptic into a list of sub-graphs for lifelong learning and adjust the train / val / test split
Task    Snapshots
T1      1...4
T2      5...9
T3      10...14
T4      15...19
...
T10     45...49
"""


def get_elliptic_temporal_as_list():
    data_list = []

    x = torch.empty([0, 165])
    y = torch.empty([0])
    edges = torch.empty([2, 0])
    timestep = torch.empty([0])

    def append_snapshot(t):
        elliptic_temp = get_elliptic_temporal(i)

        nonlocal x, y, edges
        edges = torch.cat((edges, elliptic_temp.edge_index + len(x)), 1)
        x = torch.cat((x, elliptic_temp.x))
        y = torch.cat((y, elliptic_temp.y))

    for i in range(1, 5):
        append_snapshot(i)
    train_mask, val_mask, test_mask = get_masks(len(x), 0, 0.8, 0.1, 0.1)
    data_list.append(Data(x=x, y=y, edge_index=edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                          timestep=timestep))

    for task in range(1, 10):
        prev = len(x)
        start = task * 5
        for i in range(start, start + 5):
            append_snapshot(i)
        timestep = torch.cat((timestep, torch.full((len(x) - prev,), task)))
        train_mask, val_mask, test_mask = get_masks(len(x), len(x) - prev, 0.8, 0.1, 0.1)
        data_list.append(
            Data(x=x, y=y, edge_index=edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                 timestep=timestep))
    return data_list


def get_ogbn_arxiv():
    ogbn = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/ogbn-arxiv')
    return ogbn


def get_masks(n, start, train, val, test):
    if train + val + test != 1.0:
        raise ValueError("Split must sum to 1")

    mask = torch.zeros(n)
    mask[-start:] = 1
    indices = torch.where(mask)[0]
    train_indices, other = train_test_split(indices, train_size=train, test_size=(val + test))
    val_indices, test_indices = train_test_split(other, train_size=val / (val + test),
                                                 test_size=test / (val + test))

    train_mask = torch.zeros_like(mask, dtype=torch.bool)
    val_mask = torch.zeros_like(mask, dtype=torch.bool)
    test_mask = torch.zeros_like(mask, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask
