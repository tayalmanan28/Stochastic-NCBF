import torch
import unittest
import deep_differential_network.utils as utils

class test_utils(unittest.TestCase):
    def setUp(self):
        self.f_simple = lambda x: x + 1
        self.f_list_tuple_3d = lambda x: [x + 1, x - 1]
        self.f_list_tuple_4d = lambda x: [x.unsqueeze(1) + 1, x.unsqueeze(1) - 1]
        self.f_tensor_4d = lambda x: x.unsqueeze(1) + 1

    def test_simple_tensor(self):
        x = torch.arange(0, 2048).float()
        result = utils.evaluate(self.f_simple, x)
        self.assertTrue(torch.equal(result, x + 1))

    def test_list_tuple_output_3d(self):
        x = torch.arange(0, 2048).reshape(2048, 1, 1).float()
        result = utils.evaluate(self.f_list_tuple_3d, x, n_minibatch=512)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].ndim == 3)
        self.assertTrue(torch.equal(result[0], x + 1))
        self.assertTrue(torch.equal(result[1], x - 1))

    def test_list_tuple_output_4d(self):
        x = torch.arange(0, 2048).reshape(2048, 1, 1).float()
        result = utils.evaluate(self.f_list_tuple_4d, x, n_minibatch=512)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 2)

        expected_output_0 = torch.cat([x[i:i+512].unsqueeze(1) + 1 for i in range(0, 2048, 512)], dim=1)
        print("Expected shape:", expected_output_0.shape)
        print("Actual shape:", result[0].shape)

        self.assertTrue(result[0].ndim == 4)
        self.assertTrue(torch.equal(result[0], expected_output_0), "The tensors do not match.")

    def test_tensor_output_4d(self):
        x = torch.arange(0, 2048).reshape(2048, 1, 1, 1).float()
        result = utils.evaluate(self.f_tensor_4d, x, n_minibatch=512)
        
        # Manually constructing the expected output
        expected_batches = []
        for i in range(0, x.size(0), 512):
            minibatch = x[i:i+512]
            processed = minibatch.unsqueeze(1) + 1  # Applying the same operation as f_tensor_4d
            expected_batches.append(processed)
        
        expected_result = torch.cat(expected_batches, dim=1)  # Concatenate along the axis used by evaluate
        
        self.assertTrue(result.ndim == expected_result.ndim)
        self.assertTrue(torch.equal(result, expected_result), "The result does not match the expected output.")

# Define a simple test function with known derivatives
def func(x):
    # A simple quadratic function: f(x) = (x^2).sum()
    return (x ** 2).sum()

# Known derivative of test_func with respect to x is 2x
def known_derivative(x):
    return 2 * x

# Known second derivative (Hessian) of test_func is a diagonal matrix with 2s
def known_hessian(x):
    n = x.numel()
    return torch.diag(torch.full((n,), 2))

class test_Derivatives(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.randn(10, 5, requires_grad=True)
    
    def test_scalar_output_v1(self):
    # Function with scalar output
        func = lambda x: (x ** 2).sum(dim=1)
        inputs = torch.randn(5, 3, requires_grad=True)

        # Known jacobian for func is 2*x, no need to reshape to match the output format
        known_jac = (2 * inputs).unsqueeze(1)

        jac = utils.jacobian(func, inputs, v1=True)
        
        # Ensure jac has the correct shape [5, 3] here; if not, you may need to adjust utils.jacobian or your expectations
        self.assertEqual(jac.shape, known_jac.shape, "Jacobian shape mismatch.")

        # Compare the computed Jacobian to the known gradient
        self.assertTrue(torch.allclose(jac, known_jac), "Computed Jacobian does not match the expected values.")
    
    def test_vector_output_v2(self):
        # Function with vector output
        func = lambda x: x ** 2
        inputs = torch.randn(5, 3, requires_grad=True)

        # Known jacobian for func is 2*x on the diagonal for each input
        # Create a batch of diagonal matrices with 2*x on the diagonals
        known_jac = torch.zeros(inputs.shape[0], inputs.shape[1], inputs.shape[1])
        for i in range(inputs.shape[0]):
            known_jac[i] = torch.diag(2 * inputs[i])
        known_jac = known_jac.unsqueeze(1)
        jac = utils.jacobian(func, inputs, v1=False)

        # Ensure jac has the correct shape [5, 3, 3] here; if not, you may need to adjust utils.jacobian or your expectations
        self.assertEqual(jac.shape, known_jac.shape, "Jacobian shape mismatch.")

        # Compare the computed Jacobian to the known gradient
        self.assertTrue(torch.allclose(jac, known_jac), "Computed Jacobian does not match the expected values.")

        
    def test_hessian(self):
        inputs = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
        # Compute the Hessian using the provided function
        hess = utils.hessian(func, inputs)
        # Compute the known Hessian for comparison
        expected_hess = known_hessian(inputs)
        # Verify the shape and values
        print(hess.shape)
        print(expected_hess.shape)  
        self.assertTrue(hess.shape == expected_hess.shape)
        self.assertTrue(torch.allclose(hess, expected_hess, atol=1e-4))
        
    def test_jacobian_auto(self):
        # Compute the Jacobian using the autograd function
        jac_auto = utils.jacobian_auto(func, self.inputs)
        # Compute the known derivative for comparison
        expected_jac = known_derivative(self.inputs)
        # Verify the shape and values
        self.assertTrue(jac_auto.shape == expected_jac.shape)
        self.assertTrue(torch.allclose(jac_auto, expected_jac, atol=1e-4))
        
if __name__ == '__main__':
    unittest.main()