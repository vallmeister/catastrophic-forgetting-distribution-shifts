import numpy as np

from MultiClassCSBM import MultiClassCSBM


class CSBMFeat(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.5, q_het=0.1, sigma_square=0.1, classes=20,
                 dimensions=100):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)

    def evolve(self):
        self.update_means()
        super().evolve()

    def update_means(self):
        new_means = np.zeros((self.classes, self.dimensions))
        for i in range(self.classes):
            mean, next_mean = self.means[i], self.means[(i + 1) % self.classes]
            new_mean = mean + self.sigma_square * (next_mean - mean)
            new_mean /= np.linalg.norm(new_mean)
            new_means[i] = new_mean
        self.means = new_means
