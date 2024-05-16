import csv
import logging
import os.path

import numpy as np
import pandas as pd
import torch

import node2vec_embedding
from measures import mmd_max_rbf
from node2vec_embedding import get_node2vec_embedding

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_dblp_structure_shift(p1, q1):
    dblp = torch.load('./data/real_world/dblp.pt')
    shift = []
    embedding = get_node2vec_embedding(dblp, p1, q1, length=160, k=20)
    x = embedding[(dblp.node_year <= 2004).squeeze()]
    for year in range(2004, 2016):
        mask = (dblp.node_year <= year).squeeze() if year == 2004 else (dblp.node_year == year).squeeze()
        z = embedding[mask]
        shift.append(mmd_max_rbf(x, z))
    return shift


def get_elliptic_structure_shift(p1, q1):
    elliptic = torch.load('./data/real_world/elliptic_tasks.pt')[-1]
    shift = []
    embedding = get_node2vec_embedding(elliptic, p1, q1)
    x = embedding[elliptic.t == 0]
    for task in range(10):
        z = embedding[elliptic.t == task]
        shift.append(mmd_max_rbf(x, z))
    return shift


def get_ogbn_structure_shift(p1, q1):
    ogbn = torch.load('./data/real_world/ogbn.pt')
    shift = []
    embedding = get_node2vec_embedding(ogbn, p1, q1)
    x = embedding[(ogbn.node_year <= 2011).squeeze()]
    for year in range(2011, 2021):
        mask = (ogbn.node_year <= year).squeeze() if year == 2011 else (ogbn.node_year == year).squeeze()
        z = embedding[mask]
        shift.append(mmd_max_rbf(x, z, len(x[0])))
    return shift


if __name__ == "__main__":
    logger.info('Started')
    fieldnames = ['dataset', 'p', 'q', 'avg_shift', 'max_shift']

    os.makedirs('./structure_shifts/', exist_ok=True)
    file_path = './structure_shifts/rw_structure_shift.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            logger.info('Make directory')
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    for p in node2vec_embedding.PARAMETERS:
        for q in node2vec_embedding.PARAMETERS:
            for dataset in ['dblp', 'elliptic', 'ogbn']:
                df = pd.read_csv(file_path)
                npy_name = f'./structure_shifts/{dataset}_{str(p).replace(".", "")}_{str(q).replace(".", "")}.npy'
                if ((df['dataset'] == dataset) & (df['p'] == p) & (df['q'] == q)).any():
                    logger.info(f'{dataset} with p={p} and q={q} already present')
                    continue
                elif dataset == 'dblp':
                    structure_shift = get_dblp_structure_shift(p, q)
                elif dataset == 'elliptic':
                    structure_shift = get_elliptic_structure_shift(p, q)
                elif dataset == 'ogbn':
                    structure_shift = get_ogbn_structure_shift(p, q)
                np.save(npy_name, structure_shift)
                logger.info('Numpy array saved.')
                with open(file_path, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writerow(
                        {'dataset': dataset, 'p': p, 'q': q, 'avg_shift': sum(structure_shift) / len(structure_shift),
                         'max_shift': max(structure_shift)})
                logger.info(f'Added results in CSV.\n')
    df = pd.read_csv(file_path)
    print(f'Structure shift:\n{df.to_string(index=False)}')
