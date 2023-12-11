import numpy as np

from MultiClassCSBM import MultiClassCSBM


class CSBMFeat(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.5, q_het=0.1, sigma_square=0.1, classes=20,
                 dimensions=100):
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
