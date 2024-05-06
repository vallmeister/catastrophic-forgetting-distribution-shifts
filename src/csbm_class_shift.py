import csv
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch

from csbm import CSBM_NAMES, split_static_csbm
from measures import total_variation_distance


def get_class_shift(data):
    if len(torch.unique(data.t)) == 1:
        split_static_csbm(data)
    class_shift = []
    p = Counter(data.y[data.t == 0].tolist())
    total = sum(p.values())
    for c in p.keys():
        p[c] /= total
    for task in range(10):
        q = Counter(data.y[data.t == task].tolist())
        total = sum(q.values())
        for c in q.keys():
            q[c] /= total
        class_shift.append(total_variation_distance(p, q))
    return class_shift


if __name__ == "__main__":
    fieldnames = ['dataset', 'avg_shift', 'max_shift']
    os.makedirs('./class_shifts/', exist_ok=True)
    file_path = './class_shifts/csbm_class_shift.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    df = pd.read_csv(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for dataset in CSBM_NAMES:
            for i in range(10):
                name = f'{dataset}_{i:02d}'
                if (df['dataset'] == name).any():
                    continue
                csbm = torch.load(f'./data/csbm/{name}.pt')[-1]
                shift = get_class_shift(csbm)
                np.save(f'class_shifts/{name}.npy', shift)
                writer.writerow(
                    {'dataset': name, 'avg_shift': sum(shift) / max(1, len(shift)), 'max_shift': max(shift)})
    df = pd.read_csv(file_path).round(4)
    print(f'Class shift:\n{df.to_string(index=False)}\n')
