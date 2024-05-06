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
for data in dblp_tasks:
    train, val, test = data.train_mask, data.val_mask, data.test_mask
    assert not torch.logical_and(train, val).any()
    assert not torch.logical_and(train, test).any()
    assert not torch.logical_and(val, test).any()
    print(f'X: {data.x.size()}')
    print(f'Y: {data.y.size()}, labels: {sorted(torch.unique(data.y).tolist())}')
    print(f'train mask: {train.size()}, {train.sum().item()}')
