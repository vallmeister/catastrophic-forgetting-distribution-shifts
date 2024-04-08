import numpy as np
import torch
from numpy import random
from torch_geometric.data import Data

from measures import mmd_max_rbf, total_variation_distance


class MultiClassCSBM:
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.005, q_het=0.001, sigma_square=0.1,
                 classes=16, dimensions=128):
        self.n = n
        self.classes = classes
        self.dimensions = dimensions
        self.sigma_square = sigma_square
        self.q_hom = q_hom
        self.q_het = q_het

        self.means = self.initialize_means()
        self.p = np.full((classes,), 1 / classes)

        self.X = np.empty([0, dimensions], dtype=np.float64)
        self.y = np.empty([0], dtype=np.int32)
        self.timestamps = np.empty([0], dtype=np.int32)
        self.edge_sources = []
        self.edge_targets = []

    def initialize_means(self):
        means = np.zeros((self.classes, self.dimensions))
        ones_per_mean = self.dimensions / self.classes
        curr_mean = 0
        for i in range(self.dimensions):
            if i >= (curr_mean + 1) * ones_per_mean:
                curr_mean += 1
            means[curr_mean][i] = 1.0
        for i in range(self.classes):
            means[i] = means[i] / np.linalg.norm(means[i])
        return means

    def draw_class_labels(self):
        new_labels = random.choice(list(range(self.classes)), self.n, p=self.p)
        self.y = np.concatenate((self.y, new_labels))

    def draw_node_features(self):
        cov = self.sigma_square * np.eye(self.dimensions)
        offset = len(self.X)
        node_features = np.zeros((self.n, self.dimensions))
        for i in range(offset, offset + self.n):
            class_label = self.y[i]
            class_mean = self.means[class_label]
            node_features[i - offset] = random.multivariate_normal(class_mean, cov, 1)
        self.X = np.concatenate((self.X, node_features))

    def generate_edges(self):
        end = len(self.X)
        start = end - self.n
        t = end // self.n
        q_hom = self.q_hom / t
        q_het = self.q_het / t
        for i in range(start, end):
            for j in range(end):
                if i == j:
                    continue
                self.set_edge(i, j, q_hom, q_het)

    def set_edge(self, u, v, p, q):
        if self.y[u] == self.y[v] and random.binomial(1, p):
            self.edge_sources.append(u)
            self.edge_targets.append(v)
        elif self.y[u] != self.y[v] and random.binomial(1, q):
            self.edge_sources.append(u)
            self.edge_targets.append(v)

    def get_data(self):
        edge_index = torch.tensor([self.edge_sources, self.edge_targets], dtype=torch.long)
        x = torch.tensor(self.X, dtype=torch.float)
        y = torch.tensor(self.y, dtype=torch.long)
        t = torch.tensor(self.timestamps, dtype=torch.int32)
        return Data(x=x, edge_index=edge_index, y=y, t=t)

    def evolve(self):
        self.draw_class_labels()
        self.draw_node_features()
        self.generate_edges()

    def generate_data(self, tasks=1):
        for t in range(tasks):
            self.timestamps = np.concatenate((self.timestamps, np.full((self.n,), t)))
            self.evolve()
        return self.get_data()

    def get_per_class_feature_shift_mmd_with_rbf_kernel(self):
        mmd = 0
        n = self.n
        X = self.X[:n]
        y_0 = self.y[:n]
        Z = self.X[-n:]
        y_1 = self.y[-n:]
        for c in range(self.classes):
            mmd += mmd_max_rbf(X[y_0 == c], Z[y_1 == c], self.dimensions)
        return mmd / self.classes

    def get_structure_shift_rbf_mmd(self):
        pass

    def get_class_label_shift_tvd(self):
        y_0 = self.y[:self.n]
        y_1 = self.y[-self.n:]
        P = [0] * self.classes
        Q = [0] * self.classes
        for label in y_0:
            P[label] += 1
        for label in y_1:
            Q[label] += 1
        for c in range(self.classes):
            P[c] /= self.n
            Q[c] /= self.n
        return total_variation_distance(P, Q)


class FeatureCSBM(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.005, q_het=0.001, sigma_square=0.1,
                 classes=16, dimensions=128):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)
        self.initial_means = self.means

    def evolve(self):
        super().evolve()
        self.update_means()

    def update_means(self):
        new_means = np.zeros((self.classes, self.dimensions))
        for i in range(self.classes):
            curr_mean = self.means[i]
            initial_mean, next_initial_mean = self.initial_means[i], self.initial_means[(i + 1) % self.classes]
            new_mean = curr_mean + self.sigma_square * (next_initial_mean - initial_mean)
            new_mean /= np.linalg.norm(new_mean)
            new_means[i] = new_mean
        self.means = new_means

    def get_average_pairwise_mean_distance(self):
        distance = 0
        num_pairs = 0
        for i in range(self.classes):
            for j in range(i + 1, self.classes):
                distance += np.linalg.norm(self.means[i] - self.means[j])
                num_pairs += 1
        return distance / max(1, num_pairs)

    def get_average_distance_between_curr_and_init_mean(self):
        distance = 0
        for i in range(self.classes):
            distance += np.linalg.norm(self.means[i] - self.initial_means[i])
        return distance / self.classes

    def get_average_distance_from_neighboring_means(self):
        distance = 0
        for i in range(self.classes):
            distance += np.linalg.norm(self.means[i] - self.means[(i + 1) % self.classes])
        return distance / self.classes

    def get_average_pairwise_distance_from_non_neighboring_means(self):
        distance = 0
        num_pairs = 0
        for i in range(self.classes):
            for j in range(i + 2, self.classes):
                if (j + 1) % self.classes == i:
                    continue
                distance += np.linalg.norm(self.means[i] - self.means[j])
                num_pairs += 1
        return distance / num_pairs


class StructureCSBM(MultiClassCSBM):

    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.05, q_het=0.01, sigma_square=0.1,
                 classes=16, dimensions=128):
        self.max_degree = 1
        super().__init__(n,
                         class_distribution,
                         means,
                         q_hom,
                         q_het,
                         sigma_square,
                         classes,
                         dimensions)

    def generate_edges(self):
        end = len(self.X)
        start = end - self.n
        for i in range(start, end):
            for j in range(end):
                if i == j:
                    continue
                t = j // self.n + 1
                q_hom = self.q_hom / t
                q_het = self.q_het / t
                self.set_edge(i, j, q_hom, q_het)


class ClassCSBM(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.005, q_het=0.001, sigma_square=0.1,
                 classes=16, dimensions=128):

        def rho_iter():
            yield 50
            yield 40
            yield 30
            yield 20
            yield 10
            yield 5
            yield 4
            yield 3
            yield 2
            yield 1
            yield 1

        self.imbalance_ratios = rho_iter()
        super().__init__(n, self.get_class_distribution(classes), means, q_hom, q_het, sigma_square, classes,
                         dimensions)

    def evolve(self):
        super().evolve()
        self.p = self.get_class_distribution(self.classes)

    def get_class_distribution(self, c):
        rho = next(self.imbalance_ratios)
        probabilities = np.zeros((c,))
        for k in range(c):
            probabilities[k] = (1 / rho) ** (k / (c - 1))
        s = np.sum(probabilities)
        for k in range(c):
            probabilities[k] /= s
        return probabilities


class HomophilyCSBM(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.005, q_het=0.001, sigma_square=0.1,
                 classes=16, dimensions=128):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)

    def generate_edges(self):
        end = len(self.X)
        start = end - self.n
        t = len(self.X) // self.n
        q_hom = self.q_hom / t ** 2
        q_het = self.q_het / t
        for i in range(start, end):
            for j in range(end):
                if i == j:
                    continue
                self.set_edge(i, j, q_hom, q_het)
