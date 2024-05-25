from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch_geometric
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset, EllipticBitcoinTemporalDataset


def get_dblp():
    path = Path('./data/dblp-hard')

    x = np.load(f'{path}/X.npy')
    y = np.load(f'{path}/y.npy')
    node_year = np.load(f'{path}/t.npy')
    nx_graph = nx.read_adjlist(f'{path}/adjlist.txt', nodetype=int)

    data = torch_geometric.utils.from_networkx(nx_graph)
    data.x = torch.tensor(x, dtype=torch.float)
    data.y = torch.tensor(y, dtype=torch.long)
    data.node_year = torch.tensor(node_year, dtype=torch.long)

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
    class_mask = torch.zeros(dblp.x.size(0), dtype=torch.bool)
    for cls in sorted(torch.unique(dblp.y[dblp.node_year <= 2004]).tolist()):
        class_mask |= (dblp.y == cls).squeeze()
    dblp = dblp.subgraph(class_mask)

    for year in range(2004, 2016):
        year_mask = (dblp.node_year <= year).squeeze()
        subgraph = dblp.subgraph(year_mask)
        node_mask = (subgraph.node_year <= year).squeeze() if year == 2004 else (subgraph.node_year == year).squeeze()
        train, val, test = get_mask(node_mask, seed=0)

        subgraph.train_mask = train
        subgraph.val_mask = val
        subgraph.test_mask = test

        data_list.append(subgraph)
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
    y = torch.empty([0], dtype=torch.long)
    edges = torch.empty([2, 0], dtype=torch.long)
    t = torch.empty([0], dtype=torch.long)

    def append_snapshot(time):
        elliptic_temp = get_elliptic_temporal(time)

        nonlocal x, y, edges, t
        edges = torch.cat((edges, elliptic_temp.edge_index + len(x)), 1)
        x = torch.cat((x, elliptic_temp.x))
        y = torch.cat((y, elliptic_temp.y))
        t = torch.cat((t, torch.full((len(elliptic_temp.x),), time // 5)))

    def append_data(curr_t):
        mask = (y != 2) & (t == curr_t)
        train_mask, val_mask, test_mask = get_mask(mask, seed=0)
        data_list.append(Data(x=x, y=y, edge_index=edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                              t=t))

    for task in range(10):
        start = max(1, task * 5)
        end = start + (4 if task == 0 else 5)
        for i in range(start, end):
            append_snapshot(i)
        append_data(task)
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
        node_mask = (ogbn.node_year <= year).squeeze()
        subgraph = ogbn.data.subgraph(node_mask)
        subgraph.y = subgraph.y.squeeze()
        year_mask = (subgraph.node_year <= year).squeeze() if year == 2011 else (subgraph.node_year == year).squeeze()
        train, val, test = get_mask(year_mask, seed=0)
        subgraph.train_mask = train
        subgraph.val_mask = val
        subgraph.test_mask = test
        data_list.append(subgraph)
    return data_list


def get_mask(mask, train=0.6, val=0.2, test=0.2, seed=None):
    if train + val + test != 1.0:
        raise ValueError("Split must sum to 1")

    indices = torch.where(mask)[0]
    n = len(indices)
    if seed:
        torch.manual_seed(seed)
    permuted_indices = torch.randperm(n)

    train_size = int(train * n)
    val_size = int(val * n)
    test_size = int(test * n)

    train_mask = torch.zeros_like(mask, dtype=torch.bool)
    val_mask = torch.zeros_like(mask, dtype=torch.bool)
    test_mask = torch.zeros_like(mask, dtype=torch.bool)

    train_mask[indices[permuted_indices[:train_size]]] = True
    val_mask[indices[permuted_indices[train_size:train_size + val_size]]] = True
    test_mask[indices[permuted_indices[-test_size:]]] = True

    return train_mask, val_mask, test_mask
