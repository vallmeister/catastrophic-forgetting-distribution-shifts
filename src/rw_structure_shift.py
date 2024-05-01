import csv
import threading

import torch

from measures import mmd_max_rbf
from node2vec_embedding import get_node2vec_embedding

PARAMETERS = [0.25, 0.5, 1, 2, 4]
RESULTS = {}


def dblp_structure_shift_instance(dblp, p, q):
    shift = []
    embedding = get_node2vec_embedding(dblp, p, q)
    x = embedding[(dblp.node_year <= 2004).squeeze()]
    for year in range(2004, 2016):
        mask = (dblp.node_year <= year).squeeze() if year == 2004 else (dblp.node_year == year).squeeze()
        z = embedding[mask]
        shift.append(mmd_max_rbf(x, z))
    torch.save(f'./rw_structure/dblp_{str(p).replace(".", "")}_{str(q).replace(".", "")}.pt',
               torch.tensor(shift))
    RESULTS[('ogbn', p, q)] = shift


def dblp_structure_shift():
    dblp = torch.load('./data/real_world/dblp_tasks.pt')[-1]
    threads = []
    for p in PARAMETERS:
        for q in PARAMETERS:
            t = threading.Thread(target=dblp_structure_shift_instance, args=(dblp, p, q))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()


def ogbn_structure_shift_instance(ogbn, p, q):
    shift = []
    embedding = get_node2vec_embedding(ogbn, p, q)
    x = embedding[(ogbn.node_year <= 2011).squeeze()]
    for year in range(2011, 2021):
        mask = (ogbn.node_year <= year).squeeze() if year == 2011 else (ogbn.node_year == year).squeeze()
        z = embedding[mask]
        shift.append(mmd_max_rbf(x, z))
    torch.save(f'./rw_structure/ogbn_{str(p).replace(".", "")}_{str(q).replace(".", "")}.pt',
               torch.tensor(shift))
    RESULTS[('ogbn', p, q)] = shift


def ogbn_structure_shift():
    ogbn = torch.load('./data/real_world/ogbn_tasks.pt')
    threads = []
    for p in PARAMETERS:
        for q in PARAMETERS:
            t = threading.Thread(target=ogbn_structure_shift_instance, args=(ogbn, p, q))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    t1 = threading.Thread(target=dblp_structure_shift)
    t2 = threading.Thread(target=ogbn_structure_shift)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    with open('rw_structure_shift.csv', 'w') as csv_file:
        fieldnames = ['Dataset', 'p', 'q', 'avg_shift', 'max_shift']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for k, shift_list in RESULTS.items():
            dataset, p, q = k
            writer.writerow({'Dataset': dataset, 'p': p, 'q': q, 'avg_shift': sum(shift_list) / len(shift_list),
                             'max_shift': max(shift_list)})
