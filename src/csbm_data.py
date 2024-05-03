#!/usr/bin/env python
# coding: utf-8
import concurrent.futures
# # Creating, saving and loading datasets with CSBMs

# In[1]:


import multiprocessing
import os
import torch

from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM, ClassCSBM, HomophilyCSBM

# In[2]:


n = 200

os.makedirs('./data/csbm/', exist_ok=True)


# In[3]:


def create_static_csbm(idx):
    zero_csbm = MultiClassCSBM(n=10 * n)
    zero_data = [zero_csbm.get_data()]
    torch.save(zero_data, f'./data/csbm/zero_{idx:02d}.pt')
    print(f'Done saving static_{idx:02d}')


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
    print(f'Done saving {name}_{idx:02d}')


# In[5]:

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as pool:
        #processes = []
        for i in range(10):
            pool.submit(create_static_csbm, i)
            pool.submit(create_evolving_csbm, 'base', i)
            pool.submit(create_evolving_csbm, 'feat', i)
            pool.submit(create_evolving_csbm, 'struct', i)
            pool.submit(create_evolving_csbm, 'hom', i)
            pool.submit(create_evolving_csbm, 'class', i)
        #     p = multiprocessing.Process(target=create_static_csbm, args=(i,))
        #     processes.append(p)
        #     p.start()
        #
        #     p = multiprocessing.Process(target=create_evolving_csbm, args=('base', i))
        #     processes.append(p)
        #     p.start()
        #
        #     p = multiprocessing.Process(target=create_evolving_csbm, args=('feat', i))
        #     processes.append(p)
        #     p.start()
        #
        #     p = multiprocessing.Process(target=create_evolving_csbm, args=('struct', i))
        #     processes.append(p)
        #     p.start()
        #
        #     p = multiprocessing.Process(target=create_evolving_csbm, args=('hom', i))
        #     processes.append(p)
        #     p.start()
        #
        #     p = multiprocessing.Process(target=create_evolving_csbm, args=('class', i))
        #     processes.append(p)
        #     p.start()
        # for p in processes:
        #     p.join()
