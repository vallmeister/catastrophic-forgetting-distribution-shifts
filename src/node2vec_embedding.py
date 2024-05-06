import sys

import torch
from torch_geometric.nn import Node2Vec

PARAMETERS = [0.25, 0.5, 1, 2, 4]


def get_node2vec_model(data, p, q, length=80, k=10):
    return Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=length,
        context_size=k,
        walks_per_node=10,
        num_negative_samples=1,
        p=p,
        q=q
    )


def get_node2vec_embedding(data, p, q, length=80, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4 if sys.platform == 'linux' else 0
    model = get_node2vec_model(data, p, q, length, k).to(device)
    loader = model.loader(batch_size=32, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    n = len(data.x)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.arange(n) % 2 == 0] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[torch.arange(n) % 2 == 1] = True

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
        accuracy = model.test(
            train_z=z[train_mask],
            train_y=data.y[train_mask],
            test_z=z[test_mask],
            test_y=data.y[test_mask],
            max_iter=150,
        )
        return accuracy

    for epoch in range(1, 101):
        loss = train()
        acc = test()
        if epoch % 50 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Accuracy: {acc:.3f}')

    return model.embedding.weight.cpu().detach().numpy()
