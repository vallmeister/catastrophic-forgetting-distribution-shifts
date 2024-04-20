#!/usr/bin/env python
# coding: utf-8

# # Testing basic GCN on our modified real-world tasks

# In[4]:


import torch
import torch_geometric

from datasets import get_dblp_tasks, get_elliptic_temporal_tasks, get_ogbn_arxiv_tasks


# In[ ]:


dblp = get_dblp_tasks()
elliptic = get_elliptic_temporal_tasks()
ogbn = get_ogbn_arxiv_tasks()


# In[5]:


import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# In[6]:


from measures import Result


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dblp_result = Result(dblp, GCN(len(dblp[0].x[0]), 44), device)
dblp_result.learn()
print(f'DBLP\tAP:{dblp_result.get_average_accuracy():.2f}\tAF:{dblp_result.get_average_forgetting_measure():.2f}\n')
print(dblp_result.get_result_matrix())

elliptic_result = Result(elliptic, GCN(len(elliptic[0].x[0]), 2), device)
elliptic_result.learn()
print(f'Elliptic\tAP:{elliptic_result.get_average_accuracy():.2f}\tAF:{elliptic_result.get_average_forgetting_measure():.2f}\n')
print(elliptic_result.get_result_matrix())

ogbn_result = Result(ogbn, GCN(len(ogbn[0].x[0]), torch.unique(ogbn[0].y)), device)

