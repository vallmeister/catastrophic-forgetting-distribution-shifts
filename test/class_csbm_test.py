import unittest

from src.csbm import ClassCSBM


class ClassCSBMTest(unittest.TestCase):

    def test_probabilities_always_result_in_valid_distribution(self):
        csbm_cl = ClassCSBM(n=200)
        for _ in range(10):
            self.assertEquals(sum(csbm_cl.p).round(6), 1.)
            csbm_cl.evolve()
