#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datasets as ds


# In[ ]:


dblp = ds.get_dblp()
dblp_tasks = ds.get_dblp_tasks()

elliptic = ds.get_elliptic()
elliptic_tasks = ds.get_elliptic_temporal_tasks()

ogbn = ds.get_ogbn_arxiv()
ogbn_tasks = ds.get_ogbn_arxiv_tasks()


# In[4]:


import os
import torch


# In[ ]:


os.makedirs('./data/real_world/', exist_ok=True)

torch.save(dblp, './data/real_world/dblp.pt')
torch.save(dblp_tasks, './data/real_world/dblp_tasks.pt')
        
torch.save(elliptic, './data/real_world/elliptic.pt')
torch.save(elliptic_tasks, './data/real_world/elliptic_tasks.pt')

torch.save(ogbn, './data/real_world/ogbn.pt')
torch.save(ogbn_tasks, './data/real_world/ogbn_tasks.pt')

