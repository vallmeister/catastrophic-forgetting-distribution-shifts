import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.MultiClassCSBM import MultiClassCSBM


class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        dataset = MultiClassCSBM().graph
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 20)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
