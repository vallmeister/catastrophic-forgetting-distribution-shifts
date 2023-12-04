import numpy as np
import torch
from numpy import random
from torch_geometric.data import Data


class MultiClassCSBM:
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.5, q_het=0.1, sigma_square=0.1, classes=20,
                 dimensions=100):
        self.n = n
        self.sigma_square = sigma_square
        self.classes = classes
        self.dimensions = dimensions

        self.X = np.empty([0, dimensions])
        self.y = np.empty([0], dtype=np.int32)

        if class_distribution:
            self.p = class_distribution
        else:
            self.p = np.full((classes,), 1 / classes)
        self.draw_class_labels()

        if means:
            self.means = means
        else:
            self.initialize_means()
        self.draw_node_features()

        self.q_hom = q_hom
        self.q_het = q_het

        self.edge_sources = []
        self.edge_targets = []
        self.generate_edges()

        self.data = None
        self.build_graph()
        self.tau = 1
        self.set_masks()

    # TODO: Implement Gram-Schmitt?
    def initialize_means(self):
        self.means = np.zeros((self.classes, self.dimensions))
        ones_per_mean = self.dimensions / self.classes
        curr_mean = 0
        for i in range(self.dimensions):
            if i >= (curr_mean + 1) * ones_per_mean:
                curr_mean += 1
            self.means[curr_mean][i] = 1.0
        for i in range(self.classes):
            self.means[i] = self.means[i] / np.linalg.norm(self.means[i])

    def draw_class_labels(self):
        new_labels = random.choice(list(range(self.classes)), self.n, p=self.p)
        self.y = np.concatenate((self.y, new_labels))

    def draw_node_features(self, t=0):
        cov = self.sigma_square * np.eye(self.dimensions)
        offset = t * self.n
        node_features = np.zeros((self.n, self.dimensions))
        for i in range(offset, offset + self.n):
            class_label = self.y[i]
            class_mean = self.means[class_label]
            node_features[i - offset] = random.multivariate_normal(class_mean, cov, 1)
        self.X = np.concatenate((self.X, node_features))

    def generate_edges(self, t=0):
        offset = self.n * t
        for i in range(offset, offset + self.n):
            for j in range((t + 1) * self.n):
                if i == j:
                    continue
                elif self.y[i] == self.y[j] and random.binomial(1, self.q_hom):
                    self.edge_sources.append(i)
                    self.edge_targets.append(j)
                elif self.y[i] != self.y[j] and random.binomial(1, self.q_het):
                    self.edge_sources.append(i)
                    self.edge_targets.append(j)

    def build_graph(self):
        edge_index = torch.tensor([self.edge_sources, self.edge_targets], dtype=torch.long)
        x = torch.tensor(self.X, dtype=torch.float)
        y = torch.tensor(self.y, dtype=torch.long)
        self.data = Data(x=x, edge_index=edge_index, y=y)

    def evolve(self):
        self.draw_class_labels()
        self.draw_node_features(self.tau)
        self.generate_edges(self.tau)
        self.build_graph()
        self.set_masks()
        self.tau += 1

    def set_masks(self, train=0.8, validation=0.1, test=0.1):
        num_nodes = self.n * self.tau
        self.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.train_mask[:int(train * num_nodes)] = 1
        self.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.test_mask[int(test * num_nodes):] = 1
