import csv
import os.path
from collections import Counter

import pandas as pd
import numpy as np
import torch

from measures import total_variation_distance


def get_dblp_class_shift():
    shift = []
    dblp = torch.load('./data/real_world/dblp.pt')
    observed_classes = sorted(torch.unique(dblp.y[dblp.node_year <= 2004]).tolist())
    p = Counter(dblp.y[dblp.node_year <= 2004].tolist())
    total = sum(p.values())
    for k in p.keys():
        p[k] /= total
    for year in range(2004, 2016):
        q = Counter(dblp.y[dblp.node_year <= year].tolist()) if year == 2004 else Counter(
            filter(lambda x: x in observed_classes, dblp.y[dblp.node_year == year].tolist()))
        total = sum(q.values())
        for k in q.keys():
            q[k] /= total
        shift.append(total_variation_distance(p, q))
    return shift


def get_elliptic_class_shift():
    shift = []
    elliptic = torch.load('./data/real_world/elliptic_tasks.pt')[-1]
    p = Counter(filter(lambda x: x in {0, 1}, (elliptic.y[elliptic.t == 0]).tolist()))
    total = sum(p.values())
    p[0] /= total
    p[1] /= total
    for task in range(10):
        q = Counter(filter(lambda x: x in {0, 1}, (elliptic.y[elliptic.t == task]).tolist()))
        total = sum(q.values())
        q[0] /= total
        q[1] /= total
        shift.append(total_variation_distance(p, q))
    return shift


def get_ogbn_class_shift():
    shift = []
    ogbn = torch.load('./data/real_world/ogbn.pt')
    p = Counter(ogbn.y[(ogbn.node_year <= 2011)].tolist())
    total = sum(p.values())
    for k in p.keys():
        p[k] /= total
    for year in range(2011, 2021):
        q = Counter(ogbn.y[(ogbn.node_year <= year)].tolist()) if year == 2011 else Counter(
            ogbn.y[(ogbn.node_year == year)].tolist())
        total = sum(q.values())
        for k in q.keys():
            q[k] /= total
        shift.append(total_variation_distance(p, q))
    return shift


if __name__ == "__main__":
    os.makedirs('./class_shifts/', exist_ok=True)
    file_path = './class_shifts/rw_class_shift.csv'
    fieldnames = ['dataset', 'avg_shift', 'max_shift']
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    df = pd.read_csv(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for dataset in ['dblp', 'elliptic', 'ogbn']:
            if (df['dataset'] == dataset).any():
                continue
            elif dataset == 'dblp':
                class_shift = get_dblp_class_shift()
            elif dataset == 'elliptic':
                class_shift = get_elliptic_class_shift()
            elif dataset == 'ogbn':
                class_shift = get_ogbn_class_shift()
            np.save(f'class_shifts/{dataset}.npy', class_shift)
            writer.writerow({'dataset': dataset, 'avg_shift': sum(class_shift) / max(1, len(class_shift)),
                             'max_shift': max(class_shift)})

    df = pd.read_csv(file_path).round(2)
    print(f'Class shift:\n{df.to_string(index=False)}\n')
