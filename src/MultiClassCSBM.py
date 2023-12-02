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
        self.y = np.empty([0])

        if class_distribution:
            self.p = class_distribution
        else:
            self.p = np.full((classes,), 1 / classes)
        self.y = self.draw_class_labels()

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

        self.graph = self.build_graph()

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
        return random.choice(list(range(self.classes)), self.n, p=self.p)

    def draw_node_features(self, t=0):
        cov = self.sigma_square * np.eye(self.dimensions)
        for i in range(self.n):
            class_label = self.y[i]
            class_mean = self.means[class_label]
            node_features = random.multivariate_normal(class_mean, cov, 1)
            self.X = np.concatenate((self.X, node_features))

    def generate_edges(self, t=0):
        for i in range(self.n):
            for j in range(self.n):
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
        return Data(x=x, edge_index=edge_index, y=y)

    def evolve(self):
        pass
