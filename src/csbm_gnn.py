#!/usr/bin/env python
# coding: utf-8

# # Using CSBM generated data to train a model

# In[1]:


import torch
import matplotlib.pyplot as plt
from csbm import MultiClassCSBM, FeatureCSBM, StructureCSBM


# In[2]:


n = 1600
d = 128
c = 4


# In[ ]:


csbm = FeatureCSBM(n=n, dimensions=d, classes=c)
data_list = [csbm.data]
for _ in range(9):
    csbm.evolve()
    data_list.append(csbm.data)


# In[ ]:


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


# ## Retrain model for each task

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print('\n' + 10 * '-' + ' Training and evaluating the model on each task ' + 10 * '-')
for task, data in enumerate(data_list):
    data = data.to(device)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Task {task+1:02d}, Accuracy: {acc:.4f}')


# In[ ]:


print('\n' + 10 * '-' + ' Evaluation after training the model on all tasks ' + 10 * '-')
model.eval()
for task, data in enumerate(data_list):
    data = data.to(device)
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Task {task+1:02d}, Accuracy: {acc:.4f}')


# ## Train model on T1 and evaluate on other tasks

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

data = data_list[0].to(device)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


# In[ ]:


model.eval()
for task, data in enumerate(data_list):
    data = data.to(device)
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Task {task+1:02d}, Accuracy: {acc:.4f}')

