import csv
import os

import numpy as np
import pandas as pd
import torch
from measures import mmd_max_rbf


def get_dblp_feature_shift():
    feature_shift = []
    dblp = torch.load('./data/real_world/dblp.pt')
    observed_classes = sorted(torch.unique(dblp.y[dblp.node_year <= 2004]).tolist())
    class_mask = torch.zeros_like(dblp.y)
    for c in observed_classes:
        class_mask |= (dblp.y == c)
    x = dblp.x[dblp.node_year <= 2004]
    for year in range(2004, 2016):
        mask = dblp.node_year <= year if year == 2004 else dblp.node_year == year
        z = dblp.x[class_mask & mask]
        feature_shift.append(mmd_max_rbf(x, z, len(dblp.x[0])))
    return feature_shift


def get_elliptic_feature_shift():
    feature_shift = []
    elliptic = torch.load('./data/real_world/elliptic_tasks.pt')[-1]
    class_mask = torch.zeros_like(elliptic.y)
    class_mask |= (elliptic.y == 0)
    class_mask |= (elliptic.y == 1)
    x = elliptic.x[elliptic.t == 0]
    for task in range(10):
        mask = elliptic.t == task
        z = elliptic.x[class_mask & mask]
        feature_shift.append(mmd_max_rbf(x, z, len(elliptic.x[0])))
    return feature_shift


def get_ogbn_feature_shift():
    feature_shift = []
    ogbn = torch.load('./data/real_world/ogbn.pt')
    x = ogbn.x[ogbn.node_year <= 2011]
    for year in range(2011, 2021):
        z = ogbn.x[ogbn.node_year <= year] if year == 2011 else ogbn.x[ogbn.node_year == year]
        feature_shift.append(mmd_max_rbf(x, z, len(ogbn.x[0])))
    return feature_shift


if __name__ == "__main__":
    os.makedirs('./feature_shifts/', exist_ok=True)
    file_path = './feature_shifts/rw_feature_shift.csv'
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        fieldnames = ['dataset', 'avg_shift', 'max_shift']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for dataset in ['dblp', 'elliptic', 'ogbn']:
            df = pd.read_csv(file_path)
            if (df['dataset'] == dataset).any():
                continue
            elif dataset == 'dblp':
                shift = get_dblp_feature_shift()
            elif dataset == 'elliptic':
                shift = get_elliptic_feature_shift()
            elif dataset == 'ogbn':
                shift = get_ogbn_feature_shift()
            np.save(f'./feature_shifts/{dataset}.npy')
            writer.writerow({'dataset': dataset, 'avg_shift': sum(shift) / max(1, len(shift)), 'max_shift': max(shift)})
        df = pd.read_csv(file_path).round(2)
        print(f'Feature shift:\t\t\t{df.to_string(index=False)}\n')
