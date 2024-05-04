import os

import torch

import datasets as ds

dblp = ds.get_dblp()
dblp_tasks = ds.get_dblp_tasks()

elliptic = ds.get_elliptic()
elliptic_tasks = ds.get_elliptic_temporal_tasks()

ogbn = ds.get_ogbn_arxiv()
ogbn_tasks = ds.get_ogbn_arxiv_tasks()

os.makedirs('./data/real_world/', exist_ok=True)

torch.save(dblp, './data/real_world/dblp.pt')
torch.save(dblp_tasks, './data/real_world/dblp_tasks.pt')

torch.save(elliptic, './data/real_world/elliptic.pt')
torch.save(elliptic_tasks, './data/real_world/elliptic_tasks.pt')

torch.save(ogbn, './data/real_world/ogbn.pt')
torch.save(ogbn_tasks, './data/real_world/ogbn_tasks.pt')

dblp_tasks = torch.load('./data/real_world/dblp_tasks.pt')

assert len(dblp_tasks) == 12, "Not 12 tasks"
for i in range(12):
    data_i = dblp_tasks[i]
    train_i, val_i, test_i = data_i.train_mask, data_i.val_mask, data_i.test_mask
    assert not torch.logical_and(train_i, val_i).any()
    assert not torch.logical_and(train_i, test_i).any()
    assert not torch.logical_and(val_i, test_i).any()

    for j in range(i + 1, 12):
        data_j = dblp_tasks[j]
        train_j, val_j, test_j = data_j.train_mask, data_j.val_mask, data_j.test_mask
        assert not torch.logical_and(train_i, train_j).any()
        assert not torch.logical_and(val_i, val_j).any()
        assert not torch.logical_and(test_i, test_j).any()
