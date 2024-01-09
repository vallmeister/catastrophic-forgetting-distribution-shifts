#!/usr/bin/env python
# coding: utf-8

# # Using CSBM generated data to train a model

# In[1]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from csbms import MultiClassCSBM


# In[2]:


n = 5000
d = 100
c = 20


# In[3]:


csbm = MultiClassCSBM(n=n, dimensions=d, classes=c)
data_list = []
for _ in range(10):
    data_list.append(csbm.data)
    csbm.evolve()
print(data_list)


# In[4]:


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

dataloader = DataLoader(data_list, batch_size=32)


# In[5]:


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(d, 16)
        self.conv2 = GCNConv(16, c)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# In[6]:


num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        data = batch.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


# In[7]:


model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

