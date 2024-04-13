#!/usr/bin/env python
# coding: utf-8

# # Structure shift with precompiled CSBM-data

# In[1]:


import matplotlib.pyplot as plt
import torch
from measures import mmd_max_rbf
from node2vec_embedding import max_range_node2vec_embedding


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
n = 5000
T = 10


# In[10]:


base = torch.load('./csbm/csbm_base.pt')[-1]
zero = torch.load('./csbm/csbm_zero.pt')[-1]
struct = torch.load('./csbm/csbm_struct.pt')[-1]
homophily = torch.load('./csbm/csbm_hom.pt')[-1]

names = {0: 'Base-CSBM',
         1: 'Zero-CSBM',
         2: 'Structure-CSBM',
         3: 'Homophily-CSBM'}
all_data = [base, zero, struct, homophily]


# In[11]:


for i in range(len(all_data)):
        print(f'{names[i]}'.ljust(15) + f'{len(all_data[i].edge_index[0])} edges'.rjust(15))


# In[ ]:


def get_node_embeddings(data, name):
    model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=250,
    context_size=100,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0).to(device)
    
    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=32, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    N = len(data.x)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:int(0.9 * n)] = 1
    train_mask = train_mask.repeat(N // n)
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[-int(0.1 * n):] = 1
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
    for epoch in range(100):
        loss = train()
        max_loss = max(max_loss, loss)
        acc = test()
        max_acc = max(acc, max_acc)
    print(f'{name}'.ljxust(15) + f'Loss: {max_loss:.3f}, Acc: {max_acc:.4f}')
    return model.embedding.weight.cpu().detach().numpy()


# In[ ]:


embeddings = []
for i in range(len(all_data)):
    embeddings.append(get_node_embeddings(all_data[i], names[i]))
emb_base, emb_zero, emb_struct, emb_hom = embeddings


# In[ ]:


def get_rbf_mmd(embedding):
    differences = []
    for t in range(T):
        start = t * n
        end = start + n
        differences.append(mmd_max_rbf(embedding[:n], embedding[start:end]))
    return differences


# In[ ]:


mmd_base = get_rbf_mmd(emb_base)
mmd_zero = get_rbf_mmd(emb_zero)
mmd_struct = get_rbf_mmd(emb_struct)
mmd_hom = get_rbf_mmd(emb_hom)


# In[ ]:


# plot
plt.figure(figsize=(6, 3))

plt.plot(time_steps, mmd_base, marker='o', linestyle='-', color='black', label='Base-CSBM')
plt.plot(time_steps, mmd_zero, marker='o', linestyle='-', color='gray', label='Zero-CSBM')
plt.plot(time_steps, mmd_hom, marker='o', linestyle='-', color='orange', label='Homophily-CSBM')
plt.plot(time_steps, mmd_struct, marker='o', linestyle='-', color='blue', label='Structure-CSBM')

plt.title(r'Graph structure-shift for different CSBMs')
plt.xlabel(r'$t$')
plt.ylabel(r'$MMD^{2}$ with RBF-kernel')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('structure_shift_rbf.pdf', format='pdf')
#plt.show()
plt.close()


# In[ ]:


for i in range(len(all_data)):
    print('-' * 10 + names[i] + '-' * 10)
    print(f'|V|= {all_data[i].num_nodes}, |E|= {all_data[i].num_edges}, |E|/|V| = {(all_data[i].num_edges / all_data[i].num_nodes):.1f}')
    print(f'Possible edges: {(100 * all_data[i].num_edges / all_data[i].num_nodes ** 2):.2f}%\n')

