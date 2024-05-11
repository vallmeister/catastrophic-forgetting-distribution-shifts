import concurrent.futures
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
    embedding = get_node2vec_embedding(csbm, p1, q1, 200, 60)
    x = embedding[csbm.t == 0]
    for task in range(10):
        z = embedding[csbm.t == task]
        structure_shift.append(mmd_max_rbf(x, z, len(x[0])))
        logger.info(f'Calculated shift for {task}th task.')
    return structure_shift


def calculate(idx, p1, q1, dset):
    name = f'{dset}_{idx:02d}'
    numpy_name = f'{name}_{str(p1).replace(".", "")}_{str(q1).replace(".", "")}.npy'
    print(os.listdir('./structure_shifts/'))
    if numpy_name in os.listdir('./structure_shifts/'):
        logger.info(f'File {numpy_name} already present')
        return
    logger.info(f'Saving empty list as dummy for {name}')
    np.save(f'./structure_shifts/{numpy_name}', [])
    csbm = torch.load(f'./data/csbm/{name}.pt')[-1]
    logger.info(f'Calculating structure shift for {name}...')
    shift = get_structure_shift(csbm, p1, q1)
    logger.info(f'Saving results for {name}')
    np.save(f'./structure_shifts/{numpy_name}', shift)
    logger.info(f'Results saved for {name}')


if __name__ == "__main__":
    logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('Started')

    os.makedirs('./structure_shifts/', exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for i in range(1):
            for p in PARAMETERS:
                for q in PARAMETERS:
                    for dataset in CSBM_NAMES:
                        pool.submit(calculate, i, p, q, dataset)
