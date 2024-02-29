import numpy as np
import torch
from numpy import random
from torch_geometric.data import Data

from measures import mmd_max_rbf, total_variation_distance


class MultiClassCSBM:
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.005, q_het=0.001, sigma_square=0.1,
                 classes=16, dimensions=128):
        self.n = n
        self.sigma_square = sigma_square
        self.classes = classes
        self.dimensions = dimensions

        self.X = np.empty([0, dimensions])
        self.y = np.empty([0], dtype=np.int32)
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1

        if class_distribution and len(class_distribution) == classes:
            self.p = class_distribution
        else:
            self.p = np.full((classes,), 1 / classes)
        self.draw_class_labels()

        self.initialize_means()
        self.draw_node_features()

        self.q_hom = q_hom
        self.q_het = q_het

        self.edge_sources = []
        self.edge_targets = []
        self.generate_edges()

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
        train_mask, validation_mask, test_mask = self.get_masks()
        return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=validation_mask,
                    test_mask=test_mask)

    def evolve(self):
        self.draw_class_labels()
        self.draw_node_features()
        self.generate_edges()

    def set_split(self, train, val, test):
        if train + val + test != 1.0:
            raise ValueError("Split has to sum up to 1")
        elif not 0 <= train <= 1 or not 0 <= val <= 1 or not 0 <= test <= 1:
            raise ValueError("Split-values have to be in range [0,1]")
        self.train_split, self.val_split, self.test_split = train, val, test

    def get_masks(self):
        train, val, test = self.train_split, self.val_split, self.test_split
        n = self.n
        N = len(self.X)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[-n:-int(n * (val + test))] = 1

        val_mask = torch.zeros(N, dtype=torch.bool)
        val_mask[-int((val + test) * n):-int(n * test)] = 1

        test_mask = torch.zeros(N, dtype=torch.bool)
        test_mask[-int(test * n):] = 1

        return train_mask, val_mask, test_mask

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
        self.update_means()
        super().evolve()

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


class ClassLabelCSBM(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.005, q_het=0.001, sigma_square=0.1,
                 classes=16, dimensions=128):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)

    def evolve(self):
        self.update_class_distribution()
        super().evolve()

    def update_class_distribution(self):
        N = len(self.X)
        t = N // self.n
        number_of_classes = max(self.classes - 3 * t, 1)
        probabilities = np.zeros((self.classes,))
        probabilities[:number_of_classes] = 1 / number_of_classes
        self.p = probabilities


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
