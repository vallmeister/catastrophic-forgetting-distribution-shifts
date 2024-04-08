#!/usr/bin/env python
# coding: utf-8

# # Creating, saving and loading datasets with CSBMs

# In[1]:


import numpy as np
import os
import torch

from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM, ClassCSBM, HomophilyCSBM


# In[2]:


n = 200


# In[3]:


os.makedirs('./csbm_base/', exist_ok=True)
os.makedirs('./csbm_zero/', exist_ok=True)
os.makedirs('./csbm_feat/', exist_ok=True)
os.makedirs('./csbm_struct/', exist_ok=True)
os.makedirs('./csbm_hom/', exist_ok=True)
os.makedirs('./csbm_class/', exist_ok=True)


# In[4]:


for i in range(10):
    base_csbm = MultiClassCSBM(n=n).generate_data(10)
    zero_csbm = MultiClassCSBM(n=10*n).generate_data()
    feat_csbm = FeatureCSBM(n=n).generate_data(10)
    struct_csbm = StructureCSBM(n=n).generate_data(10)
    hom_csbm = HomophilyCSBM(n=n).generate_data(10)
    class_csbm = ClassCSBM(n=n).generate_data(10)
    
    torch.save(base_csbm, f'./csbm_base/base_{i}.pt')
    torch.save(zero_csbm, f'./csbm_zero/zero_{i}.pt')
    torch.save(feat_csbm, f'./csbm_feat/feat_{i}.pt')
    torch.save(struct_csbm, f'./csbm_struct/struct_{i}.pt')
    torch.save(hom_csbm, f'./csbm_hom/hom_{i}.pt')
    torch.save(class_csbm, f'./csbm_class/class_{i}.pt')

