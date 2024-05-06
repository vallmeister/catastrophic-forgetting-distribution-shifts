#!/usr/bin/env python
# coding: utf-8

# # Testing basic GCN on our modified real-world tasks

# In[12]:


import torch

# In[2]:


dblp = torch.load('./data/real_world/dblp_tasks.pt')
elliptic = torch.load('./data/real_world/elliptic_tasks.pt')
ogbn = torch.load('./data/real_world/ogbn_tasks.pt')


# In[3]:


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


# In[4]:


from measures import Result

# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_printoptions(precision=3)

dblp_result = Result(dblp, GCN(len(dblp[0].x[0]), len(torch.unique(dblp[0].y))), device)
dblp_result.learn()
print(20 * '-')
print(f'DBLP\tAP:{dblp_result.get_average_accuracy():.3f}\tAF:{dblp_result.get_average_forgetting_measure():.3f}\n')
print(dblp_result.get_result_matrix())

print(20 * '-')
elliptic_result = Result(elliptic, GCN(len(elliptic[0].x[0]), 2), device)
elliptic_result.learn()
print(f'Elliptic\tAP:{elliptic_result.get_average_accuracy():.3f}\tAF:{elliptic_result.get_average_forgetting_measure():.3f}\n')
print(elliptic_result.get_result_matrix())

print(20 * '-')
ogbn_result = Result(ogbn, GCN(len(ogbn[0].x[0]), len(torch.unique(ogbn[0].y))), device)
ogbn_result.learn()
print(f'OGBN-arXiv\tAP:{ogbn_result.get_average_accuracy():.3f}\tAF:{ogbn_result.get_average_forgetting_measure():.3f}\n')
print(ogbn_result.get_result_matrix())

