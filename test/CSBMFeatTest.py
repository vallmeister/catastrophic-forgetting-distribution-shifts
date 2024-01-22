import unittest

import numpy as np

from src.csbms import FeatureCSBM


class CSBMFeatTest(unittest.TestCase):

    def test_new_means_are_still_unit_vectors_after_evolution(self):
        csbm_feat = FeatureCSBM(n=100)
        for _ in range(10):
            for mean in csbm_feat.means:
                self.assertEquals(np.linalg.norm(mean).round(6), 1.)
            csbm_feat.evolve()
