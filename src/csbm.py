import datetime

import numpy as np
import torch
from torch_geometric.data import Data

from datasets import get_mask
from measures import mmd_max_rbf, total_variation_distance

CSBM_NAMES = ['base', 'class', 'feat', 'hom', 'struct', 'zero']


class MultiClassCSBM:
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.0005, q_het=0.0001, sigma_square=0.1,
                 classes=16, dimensions=128):
        self.n = n
        self.classes = classes
        self.dimensions = dimensions
        self.sigma_square = sigma_square
        self.q_hom = q_hom
        self.q_het = q_het

        self.means = self.initialize_means()
        self.p = np.full((classes,), 1 / classes)

        self.x = np.empty([0, dimensions], dtype=np.float64)
        self.y = torch.empty(0, dtype=torch.long)
        self.t = torch.empty(0, dtype=torch.long)
        self.edge_sources = []
        self.edge_targets = []

        self.draw_class_labels()
        self.draw_node_features()
        self.generate_edges()

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
        new_labels = torch.tensor(np.random.choice(list(range(self.classes)), self.n, p=self.p), dtype=torch.long)
        self.y = torch.concatenate((self.y, new_labels), dim=0)

    def draw_node_features(self):
        t = len(self.x) // self.n
        curr_t = torch.full((self.n,), t)
        self.t = torch.concatenate((self.t, curr_t), dim=0)

        cov = self.sigma_square * np.eye(self.dimensions)
        offset = len(self.x)
        node_features = np.zeros((self.n, self.dimensions))
        for i in range(offset, offset + self.n):
            class_label = self.y[i]
            class_mean = self.means[class_label]
            node_features[i - offset] = np.random.multivariate_normal(class_mean, cov, 1)
        self.x = np.concatenate((self.x, node_features))

    def generate_edges(self):
        end = len(self.x)
        start = end - self.n
        for i in range(start, end):
            self.generate_homophile_edges(i)
            self.generate_heterophile_edges(i)

    def generate_homophile_edges(self, source):
        n_hom = np.random.binomial(self.n, self.q_hom)
        intra_class_mask = self.y == self.y[source]
        intra_class_mask[source] = False
        self.set_edges(intra_class_mask, n_hom, source)

    def generate_heterophile_edges(self, source):
        n_het = np.random.binomial(self.n, self.q_het)
        inter_class_mask = self.y != self.y[source]
        self.set_edges(inter_class_mask, n_het, source)

    def set_edges(self, intra_class_mask, num, source):
        indices = torch.where(intra_class_mask)[0]
        m = indices.size()[0]
        num = min(num, m)
        permuted_indices = torch.randperm(m)[:num]
        self.edge_sources.extend([source] * num)
        self.edge_targets.extend(indices[permuted_indices].tolist())

    def get_data(self):
        edge_index = torch.tensor([self.edge_sources, self.edge_targets], dtype=torch.long)
        x = torch.tensor(self.x, dtype=torch.float)
        mask = (self.t == self.t[-1])
        train_mask, val_mask, test_mask = get_mask(mask)
        return Data(x=x, edge_index=edge_index, y=self.y, t=self.t, train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask)

    def evolve(self):
        self.draw_class_labels()
        self.draw_node_features()
        self.generate_edges()

    def get_per_class_feature_shift_mmd_with_rbf_kernel(self):
        mmd = 0
        n = self.n
        X = self.x[:n]
        y_0 = self.y[:n]
        Z = self.x[-n:]
        y_1 = self.y[-n:]
        for c in range(self.classes):
            mmd += mmd_max_rbf(X[y_0 == c], Z[y_1 == c], self.dimensions)
        return mmd / self.classes

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
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.0005, q_het=0.0001, sigma_square=0.1,
                 classes=16, dimensions=128):
        super().__init__(n,
                         class_distribution,
                         means,
                         q_hom,
                         q_het,
                         sigma_square,
                         classes,
                         dimensions)
        self.initial_means = self.means.copy()

    def evolve(self):
        self.update_means()
        super().evolve()

    def update_means(self):
        new_means = np.zeros((self.classes, self.dimensions))
        for i in range(self.classes):
            curr_mean = self.means[i]
            initial_mean, next_initial_mean = self.initial_means[i], self.initial_means[(i + 1) % self.classes]
            new_mean = curr_mean + 0.1 * (next_initial_mean - initial_mean)
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

    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.0005, q_het=0.0001, sigma_square=0.1,
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

    def evolve(self):
        self.q_hom *= 1.5
        self.q_het *= 1.5
        super().evolve()


class ClassCSBM(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.0005, q_het=0.0001, sigma_square=0.1,
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
            while True:
                yield 1

        self.imbalance_ratios = rho_iter()
        super().__init__(n,
                         self.get_class_distribution(classes),
                         means,
                         q_hom,
                         q_het,
                         sigma_square,
                         classes,
                         dimensions)

    def evolve(self):
        self.p = self.get_class_distribution(self.classes)
        super().evolve()

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
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.0005, q_het=0.0001, sigma_square=0.1,
                 classes=16, dimensions=128):
        super().__init__(n,
                         class_distribution,
                         means,
                         q_hom,
                         q_het,
                         sigma_square,
                         classes,
                         dimensions)

    def evolve(self):
        diff = 0.1 * self.q_hom
        self.q_hom -= diff
        self.q_het += diff
        super().evolve()


def split_static_csbm(csbm):
    torch.manual_seed(0)
    indices = torch.randperm(len(csbm.t))
    split_indices = torch.chunk(indices, 10)
    labels = torch.zeros_like(csbm.t, dtype=torch.long)
    for i, part in enumerate(split_indices):
        labels[part] = i
    csbm.t = labels
