#!/usr/bin/env python
# coding: utf-8

# # Creating, saving and loading datasets with CSBMs

# In[1]:


import multiprocessing
import os
import torch

from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM, ClassCSBM, HomophilyCSBM

# In[2]:


n = 50

os.makedirs('./data/csbm/', exist_ok=True)


# In[3]:


def create_static_csbm(idx):
    zero_csbm = MultiClassCSBM(n=10 * n)
    zero_data = [zero_csbm.get_data()]
    torch.save(zero_data, f'./data/csbm/zero_{idx:02d}.pt')
    print(f'Done saving static_{idx}')


# In[4]:


def create_evolving_csbm(name, idx):
    if name == 'base':
        csbm = MultiClassCSBM(n=n)
    elif name == 'struct':
        csbm = StructureCSBM(n=n)
    elif name == 'feat':
        csbm = FeatureCSBM(n=n)
    elif name == 'hom':
        csbm = HomophilyCSBM(n=n)
    elif name == 'class':
        csbm = ClassCSBM(n=n)
    data_list = [csbm.get_data()]
    for _ in range(9):
        csbm.evolve()
        data_list.append(csbm.get_data())
    torch.save(data_list, f'./data/csbm/{name}_{idx:02d}.pt')
    print(f'Done saving {name}')


# In[5]:

if __name__ == "__main__":
    processes = []
    for i in range(10):
        p = multiprocessing.Process(target=create_static_csbm, args=(i,))
        p.start()

        p = multiprocessing.Process(target=create_evolving_csbm, args=('base', i))
        processes.append(p)
        p.start()

        p = multiprocessing.Process(target=create_evolving_csbm, args=('feat', i))
        processes.append(p)
        p.start()

        p = multiprocessing.Process(target=create_evolving_csbm, args=('struct', i))
        processes.append(p)
        p.start()

        p = multiprocessing.Process(target=create_evolving_csbm, args=('hom', i))
        processes.append(p)
        p.start()

        p5 = multiprocessing.Process(target=create_evolving_csbm, args=('class', i))
        p5.start()

        print(i)
        for p in processes:
            p.join()
