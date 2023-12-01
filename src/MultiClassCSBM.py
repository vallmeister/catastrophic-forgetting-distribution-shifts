import numpy as np
import torch
from torch_geometric.data import Data


class MultiClassCSBM:
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.5, q_het=0.1, sigma_square=0.1, classes=20,
                 dimensions=100):
        self.n = n

        if class_distribution:
            self.p = class_distribution
        else:
            self.p = np.full((1, classes), 1 / classes)

        if means:
            self.means = means
        else:
            self.initialize_means(classes, dimensions)
        self.q_hom = q_hom
        self.q_het = q_het
        self.sigma_square = sigma_square

    def generate_graph(self):
        pass

    # TODO: Write tests that each mean has unit length and means are pairwise equidistant
    # TODO: Implement Gram-Schmitt?
    def initialize_means(self, classes, dimensions):
        self.means = np.zeros((classes, dimensions))
        ones_per_mean = dimensions / classes
        curr_mean = 0
        for i in range(dimensions):
            if i >= (curr_mean + 1) * ones_per_mean:
                curr_mean += 1
            self.means[curr_mean][i] = 1.0
        for i in range(classes):
            self.means[i] = self.means[i] / np.linalg.norm(self.means[i])
