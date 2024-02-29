#!/usr/bin/env python
# coding: utf-8

# # Creating, saving and loading datasets with CSBMs

# In[2]:


import numpy as np
import os
import torch

from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM, ClassLabelCSBM, HomophilyCSBM


# In[3]:


torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)

n = 50

csbm_base = MultiClassCSBM(n=n)
csbm_zero = MultiClassCSBM(n=10*n)
csbm_feat = FeatureCSBM(n=n)
csbm_struct = StructureCSBM(n=n)
csbm_hom = HomophilyCSBM(n=n)
csbm_class = ClassLabelCSBM(n=n)


# In[5]:


base_dl = [csbm_base.get_data()]
zero_dl = [csbm_zero.get_data()]
feat_dl = [csbm_struct.get_data()]
struct_dl = [csbm_struct.get_data()]
hom_dl = [csbm_hom.get_data()]
class_dl = [csbm_class.get_data()]


# In[6]:


for i in range(9):
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


# In[7]:


os.makedirs('./csbm/', exist_ok=True)

torch.save(base_dl, './csbm/csbm_base.pt')
torch.save(zero_dl, './csbm/csbm_zero.pt')
torch.save(feat_dl, './csbm/csbm_feat.pt')
torch.save(struct_dl, './csbm/csbm_struct.pt')
torch.save(hom_dl, './csbm/csbm_hom.pt')
torch.save(class_dl, './csbm/csbm_class.pt')

