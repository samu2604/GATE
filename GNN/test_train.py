import unittest
import torch
from torch_geometric.data.data import Data
import train
import numpy as np
import models

class TestPositiveNegativeSplitter(unittest.TestCase):
    def runTest(self):
        mask = np.array([True, False, True, False]).astype(bool)
        data = Data()
        data.y = torch.tensor([1, 1, 0, 0]).long()
        negatives_mask, positives_mask = train.positive_negatives_splitter(mask, mask, data.y.numpy(), False) 
        self.assertListEqual(negatives_mask.tolist(), [False, False, True, False])
        self.assertListEqual(positives_mask.tolist(), [True, False, False, False])

#return_new_mask(mask, data, imbalance_factor: int):
class TestReturnNewMask(unittest.TestCase):
    def runTest(self):
        mask = np.array([True, False, True, False, False, True]).astype(bool)
        data = Data()
        data.y = torch.tensor([1, 1, 0, 0, 1, 0]).long()
        data.y_drug_target = torch.tensor([1, 1, 0, 0, 1, 0]).long()
        new_mask = train.return_new_mask(mask, data, 2)
        self.assertListEqual(new_mask.tolist(), [True, False, True, False, False, True])   

# test page rank from models.py
class TestPageRank(unittest.TestCase):
    edge_index = torch.tensor([
        [0, 0, 1, 1, 1, 2, 3, 3, 3, 5],  # source nodes, rows of the adjacency matrix
        [1, 3, 0, 2, 3, 1, 0, 1, 5, 3]   # target nodes, columns of the adjacency matrix
    ], dtype=torch.long)

    # All edges have the same weight for simplicity
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    # Create the Data object
    data = Data(edge_index=edge_index, edge_weight=edge_weight)

    ranks = models.page_rank(data, teleport_probs=0.1, damping_factor=0.85, max_iterations=10000, tol=1e-9)
    def runTest(self):
        self.assertAlmostEqual(self.ranks.tolist()[1], self.ranks.tolist()[3], places=5)
        self.assertAlmostEqual(self.ranks.tolist()[2], self.ranks.tolist()[5], places=5)
        self.assertGreater(self.ranks.tolist()[0], self.ranks.tolist()[2])
        self.assertGreater(self.ranks.tolist()[1], self.ranks.tolist()[0])
        self.assertGreater(self.ranks.tolist()[2], self.ranks.tolist()[4])

if __name__ == '__main__':
    unittest.main()
