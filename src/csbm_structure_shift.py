import csv
import logging
import os

import pandas as pd
import torch
import numpy as np

from csbm import CSBM_NAMES, split_static_csbm
from measures import mmd_max_rbf
from node2vec_embedding import PARAMETERS, get_node2vec_embedding

logger = logging.getLogger(__name__)


def get_structure_shift(csbm, p1, q1):
    logger.info(f'Starting calculation with p={p1} and q={q1}')
    if len(torch.unique(csbm.t)) == 1:
        split_static_csbm(csbm)
        logger.info(f'Added artificial split for static CSBM')
    structure_shift = []
    embedding = get_node2vec_embedding(csbm, p1, q1, 200, 80)
    x = embedding[csbm.t == 0]
    for task in range(10):
        z = embedding[csbm.t == task]
        structure_shift.append(mmd_max_rbf(x, z, len(x[0])))
        logger.info(f'Calculated shift for {task}th task.')
    return structure_shift


if __name__ == "__main__":
    logging.basicConfig(filename='log.log', level=logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('Started')
    fieldnames = ['dataset', 'p', 'q', 'avg_shift', 'max_shift']

    os.makedirs('./structure_shifts/', exist_ok=True)
    file_path = './structure_shifts/csbm_structure_shift.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()
            logger.info('CSV-file created.')

    for i in range(1):
        for p in PARAMETERS:
            for q in PARAMETERS:
                for dataset in CSBM_NAMES:
                    name = f'{dataset}_{i:02d}'
                    df = pd.read_csv(file_path)
                    if ((df['dataset'] == name) & (df['p'] == p) & (df['q'] == q)).any():
                        logger.info(f'{dataset} with  p={p} and q={q} already processed')
                        continue
                    csbm = torch.load(f'./data/csbm/{name}.pt')[-1]
                    logger.info(f'Calculating structure shift for {name}...')
                    shift = get_structure_shift(csbm, p, q)
                    logger.info(f'Saving results for {name}')
                    np.save(f'./structure_shifts/{name}_{str(p).replace(".", "")}_{str(q).replace(".", "")}.npy', shift)
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writerow({'dataset': dataset, 'p': p, 'q': q, 'avg_shift': sum(shift) / len(shift),
                                         'max_shift': max(shift)})
                        logger.info(f'Saved results for {name} with p={p} and q={q}')
    df = pd.read_csv(file_path).round(4)
    print(f'Structure shift:\n{df.to_string(index=False)}\n')
