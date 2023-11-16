import torch
from torch_geometric.data import Data




class CSBM:
    def __init__(self, n, class_distribution, means, q_hom=0.5, q_het=0.1, sigma_square=0.1):
        self.n = n
        self.p = class_distribution
        self.q_hom = q_hom
        self.q_het = q_het
        self.means = means
        self.sigma_square = sigma_square
        
    def generate_graph():
        pass
    
    def initialize_means():
        pass
