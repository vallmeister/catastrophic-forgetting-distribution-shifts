#!/usr/bin/env python
# coding: utf-8

# # Using CSBM generated data to train a model

# In[1]:


import torch
import matplotlib.pyplot as plt


# In[2]:


n = 5000
T = 10
c = 32
d = 128


# In[3]:


base_dl = torch.load('./csbm/csbm_base.pt')
zero_dl = torch.load('./csbm/csbm_zero.pt')
feat_dl = torch.load('./csbm/csbm_feat.pt')
struct_dl = torch.load('./csbm/csbm_struct.pt')
homophily_dl = torch.load('./csbm/csbm_hom.pt')
class_dl = torch.load('./csbm/csbm_class.pt')


# In[4]:


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(d, 128)
        self.conv2 = GCNConv(128, c)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# ## Retrain model for each task

# In[12]:


from measures import Result
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_printoptions(precision=2)

base_result = Result(GCN().to(device), base_dl)
zero_result = Result(GCN().to(device), zero_dl)
feat_result = Result(GCN().to(device), feat_dl)
struct_result = Result(GCN().to(device), struct_dl)
homophily_result = Result(GCN().to(device), homophily_dl)
class_result = Result(GCN().to(device), class_dl)

results = [base_result, zero_result, feat_result, struct_result, homophily_result, class_result]
names = {base_result: 'Base-CSBM',
         zero_result: 'Zero-CSBM',
         feat_result: 'Feature-CSBM',
         struct_result: 'Structure-CSBM',
         homophily_result: 'Homophily-CSBM',
         class_result: 'Class-CSBM'}
for result in results:
    result.learn()

for result in results:
    print('\n' + 10 * '=' + f' {names[result]} ' + 10 * '=')
    print(result.get_result_matrix)
    print(f'\n AP: {result.get_average_accuracy():.2f}'.ljust(10) + f'AF: {result.get_average_forgetting_measure():.2f}')

