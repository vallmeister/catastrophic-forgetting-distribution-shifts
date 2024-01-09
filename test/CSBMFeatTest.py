import unittest

import numpy as np

from src.CSBMFeat import CSBMFeat


class CSBMFeatTest(unittest.TestCase):

    def test_new_means_are_still_unit_vectors_after_evolution(self):
        csbm_feat = CSBMFeat(n=100, classes=10, dimensions=20)
        for _ in range(10):
            for mean in csbm_feat.means:
                self.assertEquals(np.linalg.norm(mean).round(6), 1.)
            csbm_feat.evolve()
