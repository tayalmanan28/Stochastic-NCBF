import torch
import unittest
from safe import calc_safe_u

class TestSafeMethods(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test, you can set up variables here
        self.x_domain = torch.randn((10, 2))
        self.h_domain = torch.randn((10, 1, 1))
        self.d_h_domain = torch.randn((10, 1, 2))
        self.d2_h_domain = torch.randn((10, 2, 2))
        self.f_x = torch.randn((10, 2, 1))
        self.g_x = torch.randn((10, 2, 1))
        self.sigma = torch.randn((2, 2))
        self.gamma = torch.randn(1)

    def test_calc_safe_u(self):
        # Test case for the calc_safe_u function
        u, l = calc_safe_u(self.x_domain, self.h_domain, self.d_h_domain, self.d2_h_domain, self.f_x, self.g_x, self.sigma, self.gamma)
        
        # Here you should add assertions to check the properties of u and l.
        # Since I don't know the expected behavior of your function, I'll just check that the output shapes are correct.
        self.assertEqual(u.shape, (10, 1, 1))
        self.assertEqual(l.shape, (10, 1, 1))

if __name__ == '__main__':
    unittest.main()