import unittest

from CSBMCl import CSBMCl


class CSBMClTest(unittest.TestCase):

    def test_probabilities_always_result_in_valid_istribution(self):
        csbm_cl = CSBMCl(n=200, classes=20, dimensions=40)
        for _ in range(10):
            self.assertEquals(sum(csbm_cl.p).round(6), 1.)
            csbm_cl.evolve()
