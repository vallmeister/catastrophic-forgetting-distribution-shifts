import math
import unittest
from numpy import linalg
from src.MultiClassCSBM import MultiClassCSBM


class MultiClassCSBMTest(unittest.TestCase):

    def test_means_are_unit_vectors(self):
        csbm = MultiClassCSBM()
        for mean in csbm.means:
            self.assertEquals(linalg.norm(mean), 1.)

    def test_means_are_pairwise_equidistant(self):
        csbm = MultiClassCSBM()
        m = len(csbm.means)
        for i in range(m):
            for j in range(i + 1, m):
                self.assertEquals(linalg.norm(csbm.means[i] - csbm.means[j]), math.sqrt(2))
