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
    data.y = torch.tensor(y)
    data.node_year = torch.tensor(node_year)

    return data


def get_dblp_tasks():
    """
    Task    Node year
    T1      until 2004
    T2      2005
    T3      2006
    ...
    T12     2015

    :return: dblp-hard sub-graphs as [Data]
    """
    data_list = []
    dblp = get_dblp()
    class_mask = torch.zeros_like(dblp.y).bool()
    for c in torch.unique(dblp.y[dblp.node_year <= 2004]).tolist():
        class_mask |= (dblp.y == c)
    for year in range(2004, 2016):
        year_mask = (dblp.node_year <= year).squeeze()
        edge_index = torch_geometric.utils.subgraph(year_mask, dblp.edge_index)[0]
        train_mask, val_mask, test_mask = get_mask(year_mask & class_mask, seed=0)
        data_list.append(Data(x=dblp.x, y=dblp.y.squeeze(), node_year=dblp.node_year, edge_index=edge_index,
                              train_mask=train_mask, val_mask=val_mask, test_mask=test_mask))
    return data_list


def get_elliptic():
    # Labels {0: licit, 1: illicit, 2: unlabeled}
    elliptic = EllipticBitcoinDataset(root='./data/Elliptic/')
    return elliptic


def get_elliptic_temporal(t):
    elliptic = EllipticBitcoinTemporalDataset(root='./data/EllipticTemporal/', t=t)
    return elliptic


def get_elliptic_temporal_tasks():
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
    data_list = []

    x = torch.empty([0, 165])
    y = torch.empty([0])
    edges = torch.empty([2, 0], dtype=torch.long)
    t = torch.empty([0])

    def append_snapshot(time):
        elliptic_temp = get_elliptic_temporal(time)

        nonlocal x, y, edges, t
        edges = torch.cat((edges, elliptic_temp.edge_index + len(x)), 1)
        x = torch.cat((x, elliptic_temp.x))
        y = torch.cat((y, elliptic_temp.y))
        t = torch.cat((t, torch.full((len(elliptic_temp.x),), time // 5)))

    def append_data():
        mask = (y == 0) | (y == 1)
        train_mask, val_mask, test_mask = get_mask(mask, seed=0)
        data_list.append(Data(x=x, y=y, edge_index=edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                              t=t))

    for task in range(10):
        start = max(1, task * 5)
        end = start + (4 if task == 0 else 5)
        for i in range(start, end):
            append_snapshot(i)
        append_data()
    return data_list


def get_ogbn_arxiv():
    ogbn = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/ogbn-arxiv')
    return ogbn


def get_ogbn_arxiv_tasks():
    """
    Task    Year
    T1      until 2011
    T2      2012
    ...
    T10     2020

    :return: OGBN-ArXiv sub-graphs as [Data]
    """
    ogbn = get_ogbn_arxiv()
    data_list = []
    for year in range(2011, 2021):
        mask = (ogbn.node_year <= year).squeeze()
        edge_index = torch_geometric.utils.subgraph(mask, ogbn.edge_index)[0]
        train, val, test = get_mask(mask, seed=0)
        data_list.append(Data(x=ogbn.x, y=ogbn.y.squeeze(), node_year=ogbn.node_year, edge_index=edge_index,
                              train_mask=train, val_mask=val, test_mask=test))
    return data_list


def get_mask(mask, train=0.8, val=0.1, test=0.1, seed=None):
    if train + val + test != 1.0:
        raise ValueError("Split must sum to 1")

    indices = torch.where(mask)[0]
    train_indices, other = train_test_split(indices, train_size=train, test_size=(val + test), random_state=seed)
    val_indices, test_indices = train_test_split(other, train_size=val / (val + test), test_size=test / (val + test),
                                                 random_state=seed)

    train_mask = torch.zeros_like(mask, dtype=torch.bool)
    val_mask = torch.zeros_like(mask, dtype=torch.bool)
    test_mask = torch.zeros_like(mask, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask
