#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from MultiClassCSBM import MultiClassCSBM
from CSBMhet import CSBMhet
from CSBMhom import CSBMhom
from metrics import mmd_linear, mmd_rbf, total_variation_distance

import torch
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import homophily


# In[2]:


dimensions = 100
gamma = 2 * dimensions
n = 2000
classes = 20
training_time = 100


# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
printnt(device)


# In[4]:


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# In[5]:


@torch.no_grad()
def test(model, data):
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


# In[6]:


def model_training(csbm):
    data = csbm.data
    model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0).to(device)

    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    accuracy = 0
    for epoch in range(training_time):
        loss = train(model, loader, optimizer)
        acc = test(model, data)
        accuracy = max(acc, accuracy)
    return model.embedding.weight.cpu().detach().numpy()


# In[7]:


time_steps = []

mmd_rbf_hom = []
mmd_rbf_het = []
mmd_rbf_const = []

mmd_linear_hom = []
mmd_linear_het = []
mmd_linear_const = []

csbm_hom = CSBMhom(n=n, dimensions=dimensions, classes=classes, q_hom=0.05)
csbm_het = CSBMhet(n=n, dimensions=dimensions, classes=classes, q_het=0.05)
csbm_const = MultiClassCSBM(n=n, dimensions=dimensions, classes=classes)

initial_embedding_hom = model_training(csbm_hom)
initial_embeddings_het = model_training(csbm_het)
initial_embeddings_const = model_training(csbm_const)

for t in range(10):
    time_steps.append(t)

    embedding_hom = model_training(csbm_hom)[:n]
    mmd_linear_hom.append(mmd_linear(initial_embedding_hom, embedding_hom))
    mmd_rbf_hom.append(mmd_rbf(initial_embedding_hom, embedding_hom, gamma))

    embedding_het = model_training(csbm_het)[:n]
    mmd_linear_het.append(mmd_linear(initial_embeddings_het, embedding_het))
    mmd_rbf_het.append(mmd_rbf(initial_embeddings_het, embedding_het, gamma))

    embedding_const = model_training(csbm_const)[:n]
    mmd_linear_const.append(mmd_linear(initial_embeddings_const, embedding_const))
    mmd_rbf_const.append(mmd_rbf(initial_embeddings_const, embedding_const, gamma))

    csbm_hom.evolve()
    csbm_het.evolve()
    csbm_const.evolve()

# plot
plt.figure(figsize=(12, 6))

plt.plot(time_steps, mmd_rbf_hom, marker='o', linestyle='-', color='b', label='CSBM-Hom')
plt.plot(time_steps, mmd_rbf_het, marker='o', linestyle='-', color='r', label='CSBM-Het')
plt.plot(time_steps, mmd_rbf_const, marker='o', linestyle='-', color='black', label='Const. CSBM')

plt.title('Graph structure shift')
plt.xlabel('Time Steps')
plt.ylabel('MMD with RBF kernel')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('structure_shift_all_rbf.pdf', format='pdf')
#plt.show()
plt.close()


# In[8]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.suptitle('Graph structure shift')

axes[0].plot(time_steps, mmd_rbf_hom, marker='o', linestyle='-', color='b', label='CSBM-Hom')
axes[1].plot(time_steps, mmd_rbf_het, marker='o', linestyle='-', color='r', label='CSBM-Het')
axes[2].plot(time_steps, mmd_rbf_const, marker='o', linestyle='-', color='black', label='Const. CSBM')
for ax in axes:
    ax.set_xlabel('Time steps')
    ax.set_ylabel('MMD with RBF kernel')
    ax.legend(loc='upper left')
    ax.grid(True)
plt.savefig('structure_shift_rbf_separate.pdf', format='pdf')
#plt.show()
plt.close()


# In[9]:


plt.figure(figsize=(12, 6))

plt.plot(time_steps, mmd_linear_hom, marker='o', linestyle='-', color='b', label='CSBM-Hom')
plt.plot(time_steps, mmd_linear_het, marker='o', linestyle='-', color='r', label='CSBM-Het')
plt.plot(time_steps, mmd_linear_const, marker='o', linestyle='-', color='black', label='Const. CSBM')

plt.title('Graph structure shift')
plt.xlabel('Time Steps')
plt.ylabel('MMD with linear kernel')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('structure_shift_linear_all.pdf', format='pdf')
#plt.show()
plt.close()


# In[10]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.suptitle('Graph structure shift')

coefficiencts_hom = np.polyfit(time_steps, mmd_linear_hom, 1)
poly_hom = np.poly1d(coefficiencts_hom)
y_fit_hom = poly_hom(time_steps)

coefficiencts_het = np.polyfit(time_steps, mmd_linear_het, 1)
poly_het = np.poly1d(coefficiencts_het)
y_fit_het = poly_hom(time_steps)

coefficiencts_const = np.polyfit(time_steps, mmd_linear_const, 1)
poly_const = np.poly1d(coefficiencts_const)
y_fit_const = poly_const(time_steps)

axes[0].plot(time_steps, mmd_linear_hom, marker='o', linestyle='-', color='b', label='CSBM-Hom')
axes[0].plot(time_steps, y_fit_hom, marker='o', linestyle='-', color='gray')
axes[1].plot(time_steps, mmd_linear_het, marker='o', linestyle='-', color='r', label='CSBM-Het')
axes[1].plot(time_steps, y_fit_het, marker='o', linestyle='-', color='gray')
axes[2].plot(time_steps, mmd_linear_const, marker='o', linestyle='-', color='black', label='Const. CSBM')
axes[2].plot(time_steps, y_fit_const, marker='o', linestyle='-', color='gray')
for ax in axes:
    ax.set_xlabel('Time steps')
    ax.set_ylabel('MMD with linear kernel')
    ax.legend(loc='upper left')
    ax.grid(True)
plt.savefig('structure_shift_linear_separate.pdf', format='pdf')
#plt.show()
plt.close()

