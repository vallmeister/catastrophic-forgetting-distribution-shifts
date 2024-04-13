#!/usr/bin/env python
# coding: utf-8

# # Creating, saving and loading datasets with CSBMs

# In[1]:


import os
import torch

from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM, ClassCSBM, HomophilyCSBM


# In[2]:


n = 50

os.makedirs('./data/csbm/', exist_ok=True)


# In[3]:


for i in range(10):
    base_csbm = MultiClassCSBM(n=n)
    base_data = [base_csbm.get_data()]
    
    zero_csbm = MultiClassCSBM(n=10*n)
    zero_data = [zero_csbm.get_data()]
    
    feat_csbm = FeatureCSBM(n=n)
    feat_data = [feat_csbm.get_data()]
    
    struct_csbm = StructureCSBM(n=n)
    struct_data = [struct_csbm.get_data()]
    
    hom_csbm = HomophilyCSBM(n=n)
    hom_data = [hom_csbm.get_data()]
    
    class_csbm = ClassCSBM(n=n)
    class_data = [class_csbm.get_data()]

    for _ in range(9):
        base_csbm.evolve()
        base_data.append(base_csbm.get_data())

        feat_csbm.evolve()
        feat_data.append(feat_csbm.get_data())

        struct_csbm.evolve()
        struct_data.append(struct_csbm.get_data())

        hom_csbm.evolve()
        hom_data.append(hom_csbm.get_data())

        class_csbm.evolve()
        class_data.append(class_csbm.get_data())
        
    
    torch.save(base_data, f'./data/csbm/base_{i}.pt')
    torch.save(zero_data, f'./data/csbm/zero_{i}.pt')
    torch.save(feat_data, f'./data/csbm/feat_{i}.pt')
    torch.save(struct_data, f'./data/csbm/struct_{i}.pt')
    torch.save(hom_data, f'./data/csbm/hom_{i}.pt')
    torch.save(class_data, f'./data/csbm/class_{i}.pt')

