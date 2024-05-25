import logging
import sys

import torch
from torch_geometric.nn import Node2Vec

PARAMETERS = [1, 2, 0.5]  # , 0.25, 4]
logger = logging.getLogger(__name__)


def get_node2vec_model(data, p, q, length, k):
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
    logger.info(f'Using device: {device}')
    num_workers = 4 if sys.platform == 'linux' else 0
    model = get_node2vec_model(data, p, q, length, k).to(device)
    logger.info(f'node2vec with walk_length={length} and context_size={k}')
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    n = len(data.x)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.arange(n) % 2 == 0] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[torch.arange(n) % 2 == 1] = True
    logger.info(f'Created masks with 50/50 split.')
    logger.info(f'Train mask: {train_mask[:10]}')
    logger.info(f'Test mask: {test_mask[:10]}')

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

    logger.info(f'Starting node2vec training')
    for epoch in range(1, 101):
        loss = train()
        acc = test()
        if epoch % 20 == 0:
            logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.2f}, Accuracy: {acc:.2f}')

    return model.embedding.weight.cpu().detach().numpy()
