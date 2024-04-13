import sys

import torch
from torch_geometric.nn import Node2Vec


def get_node2vec_model(data, p, q):
    return Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=80,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=p,
        q=q
    )


def max_range_node2vec_embedding(data):
    parameters = [0.25, 0.5, 1, 2, 4]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_acc = -1
    embedding = None
    num_workers = 4 if sys.platform == 'linux' else 0
    for p1 in parameters:
        for q1 in parameters:
            model = get_node2vec_model(data, p1, q1).to(device)
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

            curr_acc = -1
            for epoch in range(150):
                train()
                acc = test()
                curr_acc = max(curr_acc, acc)
            if curr_acc > max_acc:
                max_acc = curr_acc
                embedding = model.embedding
    return embedding.weight.cpu().detach().numpy()
