#!/usr/bin/env python
# coding: utf-8

# # Structure-shift

# We compare the following CSBMs and how their node2vec encoding changes:
# - Constant CSBM
# - CSBM-Hom
# - CSBM-Het
# - CSBM-Struct

# We find the following assumptions useful:
# - We compare nodes within the same embedding
# - We compare the embeddings of the t-th chunk of nodes against the 1st chunk
# - We use the same loader and optimizer for each epoch

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys
from csbms import MultiClassCSBM, StructureCSBM
from CSBMhet import CSBMhet
from CSBMhom import CSBMhom
from metrics import mmd_linear, mmd_max_rbf

import torch
from torch_geometric.nn import Node2Vec


# In[2]:


n = 10000
training_time = 100
T = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[3]:


csbm_zero = MultiClassCSBM(n =10*n)
csbm_constant = MultiClassCSBM(n=n)
csbm_hom = CSBMhom(n=n)
csbm_het = CSBMhet(n=n)
csbm_struct = StructureCSBM(n=n)
csbms = [csbm_constant, csbm_hom, csbm_het, csbm_struct]


# In[4]:


time_steps = list(range(T))
for _ in range(T - 1):
    for csbm in csbms:
        csbm.evolve()


# In[5]:


def get_node_embeddings(csbm):
    data = csbm.data
    model = Node2Vec(
    data.edge_index,
    embedding_dim=32,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0).to(device)
    
    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=32, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    
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
            train_z=z[data.train_mask],
            train_y=data.y[data.train_mask],
            test_z=z[data.test_mask],
            test_y=data.y[data.test_mask],
            max_iter=150,
        )
        return acc
    max_loss = 0
    max_acc = 0
    for epoch in range(training_time):
        loss = train()
        max_loss = max(max_loss, loss)
        acc = test()
        max_acc = max(acc, max_acc)
    print(f'Loss: {max_loss:.4f}, Acc: {max_acc:.4f}')
    return model.embedding.weight.cpu().detach().numpy()


# In[6]:


embeddings = []
for csbm in csbms:
    embeddings.append(get_node_embeddings(csbm))
emb_const, emb_hom, emb_het, emb_struct = embeddings
emb_zero = get_node_embeddings(csbm_zero)


# In[7]:


def get_linear_mmd(embedding):
    differences = []
    for t in range(T):
        start = t * n
        end = start + n
        differences.append(mmd_linear(embedding[:n], embedding[start:end]))
    return differences


# In[8]:


def get_rbf_mmd(embedding):
    differences = []
    for t in range(T):
        start = t * n
        end = start + n
        differences.append(mmd_max_rbf(embedding[:n], embedding[start:end]))
    return differences


# In[9]:


mmds_linear_const = get_linear_mmd(emb_const)
mmds_linear_hom = get_linear_mmd(emb_hom)
mmds_linear_het = get_linear_mmd(emb_het)
mmds_linear_struct = get_linear_mmd(emb_struct)
mmds_linear_zero = get_linear_mmd(emb_zero)

mmds_rbf_const = get_rbf_mmd(emb_const)
mmds_rbf_hom = get_rbf_mmd(emb_hom)
mmds_rbf_het = get_rbf_mmd(emb_het)
mmds_rbf_struct = get_rbf_mmd(emb_struct)
mmds_rbf_zero = get_rbf_mmd(emb_zero)


# In[10]:


# plot
plt.figure(figsize=(12, 6))

plt.plot(time_steps, mmds_linear_const, marker='o', linestyle='-', color='black', label='CSBM-Const')
plt.plot(time_steps, mmds_linear_zero, marker='o', linestyle='-', color='gray', label='CSBM-Zero')
plt.plot(time_steps, mmds_linear_hom, marker='o', linestyle='-', color='r', label='CSBM-Hom')
plt.plot(time_steps, mmds_linear_het, marker='o', linestyle='-', color='orange', label='CSBM-Het')
plt.plot(time_steps, mmds_linear_struct, marker='o', linestyle='-', color='blue', label='CSBM-Struct')

plt.title(r'Graph structure-shift for different CSBMs')
plt.xlabel('Time Steps')
plt.ylabel(r'$MMD^{2}$ with linear kernel')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('structure_shift_linear.pdf', format='pdf')
#plt.show()
plt.close()


# In[11]:


# plot
plt.figure(figsize=(12, 6))

plt.plot(time_steps, mmds_rbf_const, marker='o', linestyle='-', color='black', label='CSBM-Const')
plt.plot(time_steps, mmds_rbf_zero, marker='o', linestyle='-', color='gray', label='CSBM-Zero')
plt.plot(time_steps, mmds_rbf_hom, marker='o', linestyle='-', color='r', label='CSBM-Hom')
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


# In[12]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
fig.suptitle(r'Graph structure-shift for different CSBMs')

coefficiencts = np.polyfit(time_steps, mmds_linear_const, 1)
poly = np.poly1d(coefficiencts)
y_fit = poly(time_steps)
axes[0].plot(time_steps, mmds_linear_const, marker='o', linestyle='-', color='black', label='CSBM-Const')
axes[0].plot(time_steps, y_fit, marker='o', linestyle='-', color='gray')

coefficiencts = np.polyfit(time_steps, mmds_linear_hom, 1)
poly = np.poly1d(coefficiencts)
y_fit = poly(time_steps)
axes[1].plot(time_steps, mmds_linear_hom, marker='o', linestyle='-', color='red', label='CSBM-Hom')
axes[1].plot(time_steps, y_fit, marker='o', linestyle='-', color='gray')

coefficiencts = np.polyfit(time_steps, mmds_linear_het, 1)
poly = np.poly1d(coefficiencts)
y_fit = poly(time_steps)
axes[2].plot(time_steps, mmds_linear_het, marker='o', linestyle='-', color='orange', label='CSBM-Het')
axes[2].plot(time_steps, y_fit, marker='o', linestyle='-', color='gray')

coefficiencts = np.polyfit(time_steps, mmds_linear_struct, 1)
poly = np.poly1d(coefficiencts)
y_fit = poly(time_steps)
axes[3].plot(time_steps, mmds_linear_struct, marker='o', linestyle='-', color='b', label='CSBM-Struct')
axes[3].plot(time_steps, y_fit, marker='o', linestyle='-', color='gray')

for ax in axes:
    ax.set_xlabel('Time steps')
    ax.set_ylabel('MMD with linear kernel')
    ax.legend(loc='upper left')
    ax.grid(True)
plt.savefig('structure_shift_separate_linear.pdf', format='pdf')
#plt.show()
plt.close()


# In[13]:


for csbm in csbms:
    print(f'|V|= {csbm.data.num_nodes}, |E|= {csbm.data.num_edges}, |E|/|V| = {(csbm.data.num_edges / csbm.data.num_nodes):.1f}')
    print(f'Possible edges: {(100 * csbm.data.num_edges / csbm.data.num_nodes ** 2):.2f}%')


# In[14]:


train, validation, test = 0.8, 0.1, 0.1
n = 10
N = 20
train_mask = torch.zeros(N, dtype=torch.int)
train_mask[-n:-int(n * (validation + test))] = 1

validation_mask = torch.zeros(N, dtype=torch.int)
validation_mask[-int((validation + test) * n):-int(n * test)] = 1

test_mask = torch.zeros(n, dtype=torch.int)
test_mask[-int(test * n):] = 1

print(train_mask, validation_mask, test_mask)

