#!/usr/bin/env python
# coding: utf-8

# # Structure shift with precompiled CSBM-data

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys
from metrics import mmd_max_rbf

import torch
from torch_geometric.nn import Node2Vec


# In[13]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
n = 5000
T = 10


# In[8]:


data = torch.load('./csbm/csbm.pt')[-1]
data_het = torch.load('./csbm/csbm_het.pt')[-1]
data_struct = torch.load('./csbm/csbm_struct.pt')[-1]
data_zero = torch.load('./csbm/csbm_zero.pt')
names = {0: 'CSBM-Constant',
         1: 'CSBM-Het',
         2: 'CSBM-Struct',
         3: 'CSBM-Zero'}
all_data = [data, data_het, data_struct, data_zero]


# In[11]:


for i in range(len(all_data)):
        print(f'{names[i]}'.ljust(15) + f'{len(all_data[i].edge_index[0])} edges'.rjust(15))


# In[12]:


def get_node_embeddings(data, name):
    model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=80,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=2.0).to(device)
    
    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=32, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    N = len(data.x)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:int(0.5 * n)] = 1
    train_mask = train_mask.repeat(N // n)
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[-int(0.5 * n):] = 1
    test_mask = test_mask.repeat(N // n)
    
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(
            train_z=z[train_mask],
            train_y=data.y[train_mask],
            test_z=z[test_mask],
            test_y=data.y[test_mask],
            max_iter=150,
        )
        return acc
    max_loss = 0
    max_acc = 0
    for epoch in range(150):
        loss = train()
        max_loss = max(max_loss, loss)
        acc = test()
        max_acc = max(acc, max_acc)
    print(f'{name}'.ljust(15) + f'Loss: {max_loss:.3f}, Acc: {max_acc:.4f}')
    return model.embedding.weight.cpu().detach().numpy()


# In[ ]:


embeddings = []
for i in range(len(all_data)):
    embeddings.append(get_node_embeddings(all_data[i], names[i]))
emb_const, emb_het, emb_struct, emb_zero = embeddings


# In[14]:


def get_rbf_mmd(embedding):
    differences = []
    for t in range(T):
        start = t * n
        end = start + n
        differences.append(mmd_max_rbf(embedding[:n], embedding[start:end]))
    return differences


# In[ ]:


mmds_rbf_const = get_rbf_mmd(emb_const)
mmds_rbf_het = get_rbf_mmd(emb_het)
mmds_rbf_struct = get_rbf_mmd(emb_struct)
mmds_rbf_zero = get_rbf_mmd(emb_zero)


# In[ ]:


# plot
plt.figure(figsize=(8, 4))

plt.plot(time_steps, mmds_rbf_const, marker='o', linestyle='-', color='black', label='CSBM-Const')
plt.plot(time_steps, mmds_rbf_zero, marker='o', linestyle='-', color='gray', label='CSBM-Zero')
plt.plot(time_steps, mmds_rbf_het, marker='o', linestyle='-', color='orange', label='CSBM-Het')
plt.plot(time_steps, mmds_rbf_struct, marker='o', linestyle='-', color='blue', label='CSBM-Struct')

plt.title(r'Graph structure-shift for different CSBMs')
plt.xlabel('Time Steps')
plt.ylabel(r'$MMD^{2}$ with RBF-kernel')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('structure_shift_rbf.pdf', format='pdf')
#plt.show()
plt.close()


# In[16]:


for i in range(len(all_data)):
    print('-' * 10 + names[i] + '-' * 10)
    print(f'|V|= {all_data[i].num_nodes}, |E|= {all_data[i].num_edges}, |E|/|V| = {(all_data[i].num_edges / all_data[i].num_nodes):.1f}')
    print(f'Possible edges: {(100 * all_data[i].num_edges / all_data[i].num_nodes ** 2):.2f}%\n')

