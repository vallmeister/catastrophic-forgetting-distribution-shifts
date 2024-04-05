#!/usr/bin/env python
# coding: utf-8

# # Running experiments from .yaml and saving results in .csv

# In[1]:


import yaml
import csv
import torch
import numpy as np


# In[2]:


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# In[3]:


from measures import Result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_printoptions(precision=2)
np.set_printoptions(precision=2)

with open('experiments.yaml', 'r') as yaml_file:
    experiments = yaml.safe_load(yaml_file)

with open('results.csv', 'w', newline='') as csv_file:
    fieldnames = ['Dataset', 'AP', 'AF']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for experiment in experiments['experiments']:
        print(experiment['name'], experiment['directory'])
        for i in range(10):
            csbm = torch.load(experiment['directory'] + experiment['name'] + f'{i}.pt')
            result = Result(GCN().to(device), csbm)
            result.learn()
            ap = result.get_average_accuracy()
            af = 0 if experiment['name'] == 'zero_' else result.get_average_forgetting_measure()
            writer.writerow({'Dataset': f'{experiment["name"]}', 'AP': f'{ap}', 'AF': f'{af}'})
            print(f'AP: {ap:.2f}'.ljust(10) + f' AF: {af:.2f}')
        print()


# In[4]:


import pandas as pd

pd.set_option('dispplay.float_format', lambda x: f'{x:.3f}')
performance = pd.read_csv('results.csv')
print(performance.groupby('Dataset').mean())

