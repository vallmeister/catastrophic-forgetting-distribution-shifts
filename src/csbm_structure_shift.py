import csv
import logging
import os

import numpy as np
import pandas as pd
import torch

from csbm import CSBM_NAMES
from measures import mmd_max_rbf
from node2vec_embedding import PARAMETERS, get_node2vec_embedding

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_structure_shift(csbm, p1, q1):
    structure_shift = []
    embedding = get_node2vec_embedding(csbm, p1, q1)
    x = embedding[csbm.t == 0]
    for task in range(10):
        z = embedding[csbm.t == task]
        structure_shift.append(mmd_max_rbf(x, z, len(x[0])))
    return structure_shift


if __name__ == "__main__":
    logger.info('Started')
    fieldnames = ['dataset', 'p', 'q', 'avg_shift', 'max_shift']

    os.makedirs('./structure_shifts/', exist_ok=True)
    file_path = './structure_shifts/csbm_structure_shift.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    for i in range(2):
        for p in PARAMETERS:
            for q in PARAMETERS:
                if p in {4, 0.25} or q in {4, 0.25}:
                    continue
                for dataset in CSBM_NAMES:
                    name = f'{dataset}_{i:02d}'
                    npy_name = f'./structure_shifts/{name}_{str(p).replace(".", "")}_{str(q).replace(".", "")}.npy'
                    df = pd.read_csv(file_path)
                    if ((df['dataset'] == dataset) & (df['p'] == p) & (df['q'] == q)).any():
                        logger.info(f'{dataset} with  p={p} and q={q} already processed')
                        continue
                    csbm = torch.load(f'./data/csbm/{name}.pt')[-1]
                    logger.info(f'Structure shift: {name.ljust(12)}| p={p} | q={q} | |E|={csbm.edge_index.size(1)}')
                    shift = get_structure_shift(csbm, p, q)
                    np.save(npy_name, shift)
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writerow({'dataset': dataset, 'p': p, 'q': q, 'avg_shift': sum(shift) / len(shift),
                                         'max_shift': max(shift)})
                        logger.info(f'Results saved\n')
    df = pd.read_csv(file_path).round(4)
    print(f'Structure shift:\n{df.to_string(index=False)}\n')
