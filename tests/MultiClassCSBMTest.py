import math
import unittest
from numpy import linalg
from src.MultiClassCSBM import MultiClassCSBM


class MultiClassCSBMTest(unittest.TestCase):
    csbm = MultiClassCSBM()

    def test_means_are_unit_vectors(self):
        for mean in self.csbm.means:
            self.assertEquals(linalg.norm(mean), 1.)

    def test_means_are_pairwise_equidistant(self):
        m = len(self.csbm.means)
        for i in range(m):
            for j in range(i + 1, m):
                self.assertEquals(linalg.norm(self.csbm.means[i] - self.csbm.means[j]), math.sqrt(2))

    def test_number_of_nodes(self):
        self.assertEquals(self.csbm.data.num_nodes, 5000)

    def test_number_of_node_features(self):
        self.assertEquals(self.csbm.data.num_node_features, 100)

    def test_self_loops(self):
        self.assertFalse(self.csbm.data.has_self_loops())

    def test_is_directed(self):
        self.assertTrue(self.csbm.data.is_directed())

    # def test_num_classes(self):
    #     self.assertEquals(self.csbm.graph.num_classes, 20)

    def test_no_edges_from_old_to_new_nodes(self):
        evolving_csbm = MultiClassCSBM(n=100, classes=10, dimensions=20)
        evolving_csbm.evolve()
        for u, v in zip(evolving_csbm.edge_sources, evolving_csbm.edge_targets):
            self.assertFalse(u < 100 <= v)
