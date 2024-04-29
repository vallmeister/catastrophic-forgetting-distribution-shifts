import numpy as np
import torch
import torch.nn.functional as F

from sklearn import metrics


def mmd_linear(X, Z):
    XX = np.dot(X, X.T)
    ZZ = np.dot(Z, Z.T)
    XZ = np.dot(X, Z.T)
    return XX.mean() + ZZ.mean() - 2 * XZ.mean()


def mmd_rbf(X, Z, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    ZZ = metrics.pairwise.rbf_kernel(Z, Z, gamma)
    XZ = metrics.pairwise.rbf_kernel(X, Z, gamma)
    return XX.mean() + ZZ.mean() - 2 * XZ.mean()


def mmd_max_rbf(X, Z, d=128):
    GAMMA = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
             25.0, 50.0, 75.0, 100.0, d, 1.5 * d, 2 * d]
    max_mmd = 0
    for g in GAMMA:
        mmd = mmd_rbf(X, Z, g)
        max_mmd = max(max_mmd, mmd)
    return max_mmd


def total_variation_distance(P, Q):
    c = len(P)
    return 0.5 * round(sum(abs(P[k] - Q[k]) for k in range(c)), 4)


class Result:
    def __init__(self, data_list, gnn, device):
        self.data_list = data_list
        self.model = gnn.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=0.001)
        self.result_matrix = torch.zeros(len(data_list), len(data_list))

    def get_result_matrix(self):
        return self.result_matrix

    def train_model(self, i):
        data = self.data_list[i].to(self.device)
        self.model.train()
        for epoch in range(300):
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

    def test_model(self, i):
        data = self.data_list[i].to(self.device)
        self.model.eval()
        pred = self.model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        return int(correct) / int(data.test_mask.sum())

    def learn(self):
        for i in range(len(self.result_matrix)):
            self.train_model(i)
            for j in range(len(self.result_matrix)):
                self.result_matrix[i][j] = self.test_model(j)

    def get_average_accuracy(self):
        tasks = len(self.data_list)
        return 1 / tasks * sum(self.result_matrix[tasks - 1][i] for i in range(tasks))

    def get_forgetting_measure(self, i, j):
        return max(self.result_matrix[k][i].item() for k in range(j)) - self.result_matrix[j][i].item()

    def get_average_forgetting_measure(self):
        tasks = len(self.data_list)
        if tasks > 1:
            return 1 / (max(1, tasks - 1)) * sum(self.get_forgetting_measure(i, tasks - 1) for i in range(tasks))
        else:
            return 0
