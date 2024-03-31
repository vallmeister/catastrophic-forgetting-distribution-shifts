#!/usr/bin/env python
# coding: utf-8

# # Creating, saving and loading datasets with CSBMs

# In[1]:


import numpy as np
import os
import torch

from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM, ClassCSBM, HomophilyCSBM


# In[5]:


n = 5000


# In[3]:


os.makedirs('./csbm_base/', exist_ok=True)
os.makedirs('./csbm_zero/', exist_ok=True)
os.makedirs('./csbm_feat/', exist_ok=True)
os.makedirs('./csbm_struct/', exist_ok=True)
os.makedirs('./csbm_hom/', exist_ok=True)
os.makedirs('./csbm_class/', exist_ok=True)


# In[4]:


for i in range(10):
    csbm_base = MultiClassCSBM(n=n)
    csbm_zero = MultiClassCSBM(n=10*n)
    csbm_feat = FeatureCSBM(n=n)
    csbm_struct = StructureCSBM(n=n)
    csbm_hom = HomophilyCSBM(n=n)
    csbm_class = ClassCSBM(n=n)

    base_dl = [csbm_base.get_data()]
    zero_dl = [csbm_zero.get_data()]
    feat_dl = [csbm_struct.get_data()]
    struct_dl = [csbm_struct.get_data()]
    hom_dl = [csbm_hom.get_data()]
    class_dl = [csbm_class.get_data()]

    for t in range(9):
        csbm_base.evolve()
        base_dl.append(csbm_base.get_data())
        
        csbm_feat.evolve()
        feat_dl.append(csbm_feat.get_data())
        
        csbm_struct.evolve()
        struct_dl.append(csbm_struct.get_data())
    
        csbm_hom.evolve()
        hom_dl.append(csbm_hom.get_data())
    
        csbm_class.evolve()
        class_dl.append(csbm_class.get_data())

    torch.save(base_dl, f'./csbm_base/base_{i}.pt')
    torch.save(zero_dl, f'./csbm_zero/zero_{i}.pt')
    torch.save(feat_dl, f'./csbm_feat/feat_{i}.pt')
    torch.save(struct_dl, f'./csbm_struct/struct_{i}.pt')
    torch.save(hom_dl, f'./csbm_hom/hom_{i}.pt')
    torch.save(class_dl, f'./csbm_class/class_{i}.pt')

