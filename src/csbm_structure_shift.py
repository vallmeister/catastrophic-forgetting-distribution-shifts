import csv
import os

import pandas as pd
import torch
import numpy as np

from csbm import CSBM_NAMES, split_static_csbm
from measures import mmd_max_rbf
from node2vec_embedding import PARAMETERS, get_node2vec_embedding


def get_structure_shift(csbm, p1, q1):
    if len(torch.unique(csbm.t)) == 1:
        split_static_csbm(csbm)
    structure_shift = []
    embedding = get_node2vec_embedding(csbm, p1, q1, 250, 100)
    x = embedding[csbm.t == 0]
    for task in range(10):
        z = embedding[csbm.t == task]
        structure_shift.append(mmd_max_rbf(x, z, len(x[0])))
    return structure_shift


if __name__ == "__main__":
    fieldnames = ['dataset', 'p', 'q', 'avg_shift', 'max_shift']

    os.makedirs('./structure_shifts/', exist_ok=True)
    file_path = './structure_shifts/csbm_structure_shift.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    df = pd.read_csv(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for dataset in CSBM_NAMES:
            for i in range(1):
                for p in PARAMETERS:
                    for q in PARAMETERS:
                        name = f'{dataset}_{i:02d}'
                        if ((df['dataset'] == name) & (df['p'] == p) & (df['q'] == q)).any():
                            continue
                        csbm = torch.load(f'./data/csbm/{name}.pt')[-1]
                        shift = get_structure_shift(csbm, p, q)
                        np.save(f'./structure_shifts/{name}_{str(p).replace(".", "")}_{str(q).replace(".", "")}.npy',
                                shift)
                        writer.writerow({'dataset': dataset, 'p': p, 'q': q, 'avg_shift': sum(shift) / len(shift),
                                         'max_shift': max(shift)})
    df = pd.read_csv(file_path).round(4)
    print(f'Structure shift:\n{df.to_string(index=False)}\n')
