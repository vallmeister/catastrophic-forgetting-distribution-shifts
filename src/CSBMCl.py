import numpy as np

from csbms import MultiClassCSBM


class CSBMCl(MultiClassCSBM):
    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.5, q_het=0.1, sigma_square=0.1, classes=20,
                 dimensions=100):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)

    def evolve(self):
        self.update_class_distribution()
        super().evolve()

    def update_class_distribution(self):
        number_of_classes = max(self.classes - 2 * self.tau, 1)
        probabilities = np.zeros((self.classes,))
        probabilities[:number_of_classes] = 1 / number_of_classes
        self.p = probabilities
