from MultiClassCSBM import MultiClassCSBM


class CSBMhom(MultiClassCSBM):

    def __init__(self, n=5000, class_distribution=None, means=None, q_hom=0.05, q_het=0.1, sigma_square=0.1, classes=20,
                 dimensions=100):
        super().__init__(n, class_distribution, means, q_hom, q_het, sigma_square, classes, dimensions)

    def evolve(self):
            self.update_q_hom()
            super().evolve()

    def update_q_hom(self):
        self.q_hom += 0.1
