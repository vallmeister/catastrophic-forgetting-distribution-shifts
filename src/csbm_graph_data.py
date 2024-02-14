#!/usr/bin/env python
# coding: utf-8

# # Creating, saving and loading datasets with CSBMs

# In[2]:


import numpy as np
import os
import torch

from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM
from CSBMhet import CSBMhet


# In[3]:


torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)

n = 50

csbm = MultiClassCSBM(n=n)
csbm_zero = MultiClassCSBM(n=10*n)
csbm_het = CSBMhet(n=n)
csbm_struct = StructureCSBM(n=n)


# In[4]:


csbm_data_list = [csbm.get_data()]
csbm_het_data_list = [csbm_het.get_data()]
csbm_struct_data_list = [csbm_struct.get_data()]


# In[5]:


for i in range(9):
    csbm.evolve()
    csbm_data_list.append(csbm.get_data())
    
    csbm_het.evolve()
    csbm_het_data_list.append(csbm_het.get_data())
    
    csbm_struct.evolve()
    csbm_struct_data_list.append(csbm_struct.get_data())


# ## Simply by hand

# In[6]:


os.makedirs('./csbm/', exist_ok=True)

torch.save(csbm_zero.get_data(), './csbm/csbm_zero.pt')
torch.save(csbm_data_list, './csbm/csbm.pt')
torch.save(csbm_het_data_list, './csbm/csbm_het.pt')
torch.save(csbm_struct_data_list, './csbm/csbm_struct.pt')

