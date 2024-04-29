import unittest

import torch

import src.datasets as ds


class DatasetTest(unittest.TestCase):
    elliptic_tasks = ds.get_elliptic_temporal_tasks()

    def test_elliptic_temporal_list(self):
        self.assertEqual(len(self.elliptic_tasks), 10)
        prev = 0
        for time in range(10):
            data = self.elliptic_tasks[time]
            self.assertTrue(prev < len(data.x))

            prev = len(data.x)
            unique_timestep = torch.unique(data.t)

            self.assertTrue(len(unique_timestep) == time + 1)
            self.assertTrue(max(unique_timestep) == time)
            self.assertTrue(min(unique_timestep) == 0)

            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask

            self.assertFalse(torch.any(train_mask & val_mask).item())
            self.assertFalse(torch.any(train_mask & test_mask).item())
            self.assertFalse(torch.any(val_mask & test_mask).item())

            unlabeled = data.y == 2
            self.assertFalse(torch.any(train_mask & unlabeled).item())
            self.assertFalse(torch.any(val_mask & unlabeled).item())
            self.assertFalse(torch.any(test_mask & unlabeled).item())

    def test_elliptic_and_elliptic_temporal_list_equal_graphs(self):
        elliptic = ds.get_elliptic()
        temp = self.elliptic_tasks[-1]

        self.assertTrue(torch.equal(elliptic.x, temp.x))
        self.assertTrue(torch.equal(elliptic.y, temp.y))
        self.assertTrue(torch.equal(elliptic.edge_index, temp.edge_index))

    def test_ogbn_list(self):
        ogbn_tasks = ds.get_ogbn_arxiv_tasks()
        self.assertEqual(len(ogbn_tasks), 10)

        for i in range(10):
            for j in range(i + 1, 10):
                train_i, train_j = ogbn_tasks[i].train_mask, ogbn_tasks[j].train_mask
                val_i, val_j = ogbn_tasks[i].val_mask, ogbn_tasks[j].val_mask
                test_i, test_j = ogbn_tasks[i].test_mask, ogbn_tasks[j].test_mask

                self.assertFalse(torch.logical_and(train_i, val_i).any())
                self.assertFalse(torch.logical_and(train_i, test_i).any())
                self.assertFalse(torch.logical_and(val_i, test_i).any())

                self.assertFalse(torch.logical_and(train_i, train_j).any())
                self.assertFalse(torch.logical_and(val_i, val_j).any())
                self.assertFalse(torch.logical_and(test_i, test_j).any())

    def test_dblp_list(self):
        dblp_tasks = ds.get_dblp_tasks()
        self.assertEqual(len(dblp_tasks), 12)
