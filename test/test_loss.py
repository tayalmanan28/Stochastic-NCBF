import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from loss import *

class SimpleModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)


model = SimpleModel(input_dim=3, output_dim=1)  # Example model

class TestLipschitzFunction(unittest.TestCase):
    def test_single_layer(self):
        model = SimpleModel(input_dim=3, output_dim=1)  # SimpleModel is adjusted to have only one linear layer
        lambdas = torch.tensor([1.0])
        lip = 0.9
        expected_output = torch.tensor([[1.0]])  # Adjust based on expected behavior
        
        actual_output = lipschitz(lambdas, lip, model)
        
        self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-4))
        
    def test_zero_lambdas(self):
        model = SimpleModel(input_dim=3, output_dim=1)
        lambdas = torch.zeros(1)  # Assuming lambdas match the number of layers or other criteria
        lip = 0.9
        expected_output = torch.tensor([[0.0]])  # Hypothetical expected output
        
        actual_output = lipschitz(lambdas, lip, model)
        
        self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-4))
    
    
if __name__ == '__main__':
    unittest.main()