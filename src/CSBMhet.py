from csbms import MultiClassCSBM


class CSBMhet(MultiClassCSBM):

    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.05, q_het=0.01, sigma_square=0.1,
                 classes=16, dimensions=128):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)

    def evolve(self):
        self.update_q_het()
        super().evolve()

    def update_q_het(self):
        self.q_het = min(0.5, self.q_het + 0.05)
