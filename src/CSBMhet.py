from csbms import MultiClassCSBM
from numpy import random


class CSBMhet(MultiClassCSBM):

    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.005, q_het=0.001, sigma_square=0.1,
                 classes=16, dimensions=128):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)

    def evolve(self):
        super().evolve()

    def generate_edges(self):
        t = len(self.X) // self.n
        q_het = min(self.q_het * t, 0.5)
        end = len(self.X)
        start = end - self.n
        for i in range(start, end):
            for j in range(end):
                if i == j:
                    continue
                if self.y[i] == self.y[j] and random.binomial(1, self.q_hom):
                    self.edge_sources.append(i)
                    self.edge_targets.append(j)
                elif self.y[i] != self.y[j] and random.binomial(1, q_het):
                    self.edge_sources.append(i)
                    self.edge_targets.append(j)
