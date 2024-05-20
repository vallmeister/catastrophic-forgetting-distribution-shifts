import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(feature_dim, 128)
        self.conv2 = GCNConv(128, num_classes)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.001)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def observe(self, data):
        self.opt.zero_grad()
        train_mask = data.train_mask
        out = self(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        self.opt.step()
