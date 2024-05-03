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
    mask = dblp.node_year <= 2004
    for year in range(2004, 2016):
        year_mask = dblp.node_year <= year if year == 2004 else dblp.node_year == year
        mmd = 0
        for c in observed_classes:
            class_mask = dblp.y == c
            x = dblp.x[mask & class_mask]
            z = dblp.x[year_mask & class_mask]
            mmd += mmd_max_rbf(x, z, len(x[0]))
        feature_shift.append(mmd / len(observed_classes))
    return feature_shift


def get_elliptic_feature_shift():
    feature_shift = []
    elliptic = torch.load('./data/real_world/elliptic_tasks.pt')[-1]
    for task in range(10):
        t_mask = elliptic.t == task
        mmd = 0
        for c in [0, 1]:
            class_mask = elliptic.y == c
            x = elliptic.x[(elliptic.t == 0) & class_mask]
            z = elliptic.x[t_mask & class_mask]
            mmd += mmd_max_rbf(x, z, len(x[0]))
        feature_shift.append(mmd / 2)
    return feature_shift


def get_ogbn_feature_shift():
    feature_shift = []
    ogbn = torch.load('./data/real_world/ogbn.pt')
    observed_classes = sorted(torch.unique(ogbn.y).tolist())
    for year in range(2011, 2021):
        year_mask = ogbn.node_year <= year if year == 2011 else ogbn.node_year == year
        mmd = 0
        for c in observed_classes:
            class_mask = ogbn.y == c
            x = ogbn.x[(ogbn.node_year <= 2011) & class_mask]
            z = ogbn.x[year_mask & class_mask]
            mmd += mmd_max_rbf(x, z, len(x[0]))
        feature_shift.append(mmd / len(observed_classes))
    return feature_shift


if __name__ == "__main__":
    os.makedirs('./feature_shifts/', exist_ok=True)
    file_path = './feature_shifts/rw_feature_shift.csv'
    fieldnames = ['dataset', 'avg_shift', 'max_shift']
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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
    print(f'Feature shift:\n{df.to_string(index=False)}\n')
