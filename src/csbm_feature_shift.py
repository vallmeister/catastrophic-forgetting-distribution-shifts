import csv
import os
import numpy as np
import pandas as pd
import torch

from measures import mmd_max_rbf


def get_feature_shift(csbm):
    feature_shift = []
    for task in range(torch.max(csbm.t).item() + 1):
        t_mask = csbm.t == task
        mmd = 0
        observed_classes = sorted(torch.unique(csbm.y).tolist())
        for c in observed_classes:
            class_mask = csbm.y == c
            x = csbm.x[(csbm.t == 0) & class_mask]
            z = csbm.x[t_mask & class_mask]
            mmd += mmd_max_rbf(x, z, len(x[0]))
        feature_shift.append(mmd / len(observed_classes))
    return feature_shift


if __name__ == "__main__":
    csbm_names = ['base', 'class', 'feat', 'hom', 'struct', 'zero']
    fieldnames = ['dataset', 'avg_shift', 'max_shift']
    os.makedirs('./feature_shifts/', exist_ok=True)
    file_path = './feature_shifts/csbm_feature_shift.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    df = pd.read_csv(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for dataset in csbm_names:
            for i in range(10):
                name = f'{dataset}_{i:02d}'
                if (df['dataset'] == name).any():
                    continue
                csbm = torch.load(f'./data/csbm/{name}.pt')[-1]
                shift = get_feature_shift(csbm)
                np.save(f'./feature_shifts/{name}.npy', shift)
                writer.writerow(
                    {'dataset': name, 'avg_shift': sum(shift) / max(1, len(shift)), 'max_shift': max(shift)})
    df = pd.read_csv(file_path).round(4)
    print(f'Feature shift:\n{df.to_string(index=False)}\n')
