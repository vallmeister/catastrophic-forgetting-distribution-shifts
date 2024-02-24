#!/usr/bin/env python
# coding: utf-8

# # Using CSBM generated data to train a model

# In[2]:


import torch
import matplotlib.pyplot as plt


# In[6]:


n = 5000
T = 10
c = 32
d = 128


# In[7]:


data_list_constant = torch.load('./csbm/csbm.pt')
data_list_struct = torch.load('./csbm/csbm_struct.pt')


# In[8]:


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

# In[9]:


def train_from_1_to_t(model, data_list):
    print('\n' + 10 * '-' + ' Training and evaluating the model on each task ' + 10 * '-')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    results = []
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
        results.append(acc)
    return results


# In[12]:


from metrics import Result
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

time_steps = list(range(T))
model_constant = GCN().to(device)
model_struct = GCN().to(device)
result_constant = Result(model_constant, data_list_constant)
result_struct = Result(model_struct, data_list_struct)

result_constant.learn()
result_struct.learn()

print(f'{result_constant.get_result_matrix():.3f}')
print(result_constant.get_result_matrix())

