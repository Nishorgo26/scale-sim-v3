"""Small tests for the Project 3 shifter arithmetic."""
import unittest
import numpy as np
from .shiftmac import apot_quantize, matmul, mitchell_mul, pot_quantize


# Basic NumPy tests for the different MAC types
class ArithmeticTests(unittest.TestCase):
    # Zero weights should stay zero after quantizing
    # This also checks that log2(0) does not cause problems
    def test_zero_weights(self):
        z = np.zeros((3, 4))
        np.testing.assert_array_equal(pot_quantize(z), z)
        np.testing.assert_array_equal(apot_quantize(z), z)

    # Powers of two should not change when using PoT
    # Mitchell should return zero for zero input and keep the correct sign
    def test_signs_and_zeros(self):
        w = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        np.testing.assert_allclose(pot_quantize(w), w)
        self.assertEqual(float(mitchell_mul(0, -3)), 0.0)
        self.assertLess(float(mitchell_mul(-1.3, 2.7)), 0)

    # Every MAC option should return the same output dimensions
    # A 1x2 matrix times a 2x1 matrix should give a 1x1 result
    def test_matrix_shapes(self):
        a = np.array([[1.0, -2.0]])
        w = np.array([[2.0], [4.0]])
        for pe in ("exact", "pot", "apot", "mitchell"):
            self.assertEqual(matmul(a, w, pe=pe).shape, (1, 1))

    # Matrix multiplication should reject dimensions that do not match
    def test_invalid_matrix(self):
        with self.assertRaises(ValueError):
            matmul(np.zeros((2, 3)), np.zeros((4, 2)))

    # PoT needs more than one bit in this model
    # An invalid bit count should raise an error
    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            pot_quantize(np.ones(2), bits=1)


# Tests for the PyTorch version of Mitchell multiplication
class TorchLayerTests(unittest.TestCase):
    # Load PyTorch and the custom layers once for these tests
    # Skip this group if the required packages are not installed
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from .mitchell_layers import MitchellConv2d, MitchellLinear, mitchell_torch
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.torch = torch
        cls.MitchellConv2d = MitchellConv2d
        cls.MitchellLinear = MitchellLinear
        cls.mitchell_torch = staticmethod(mitchell_torch)

    # The PyTorch version should match our original NumPy function
    # The inputs include positive, negative, and zero values
    def test_mitchell_torch_matches_numpy(self):
        x = np.array([-3.2, 0, 2.5])
        y = np.array([2.6, 5, -4.1])
        got = self.mitchell_torch(self.torch.tensor(x), self.torch.tensor(y)).numpy()
        np.testing.assert_allclose(got, mitchell_mul(x, y), atol=1e-10)

    # Mitchell should match normal Linear when the weights are powers of two
    # This also checks that the bias is included in the result
    def test_linear_powers_of_two(self):
        torch = self.torch
        layer = torch.nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[2.0, -4.0]]))
            layer.bias.fill_(1.0)
        x = torch.tensor([[1.0, 3.0]])
        got = self.MitchellLinear(layer)(x)
        torch.testing.assert_close(got, layer(x))

    # Mitchell should match normal Conv2d in this simple power-of-two case
    # Check every pixel in the output, not just its shape
    def test_conv_powers_of_two(self):
        torch = self.torch
        layer = torch.nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(2.0)
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        torch.testing.assert_close(self.MitchellConv2d(layer)(x), layer(x))

    def test_optimized_matrix_and_conv_match_original_chunking(self):
        torch = self.torch
        from .mitchell_layers import mitchell_matrix
        from torch.nn import functional as F

        def original_matrix(a, w):
            outputs, positions = w.shape[0], a.shape[1]
            result = a.new_zeros((outputs, positions))
            for o in range(0, outputs, 8):
                for p in range(0, positions, 16):
                    part = result[o:o + 8, p:p + 16]
                    for kk in range(0, a.shape[0], 64):
                        aa = a[kk:kk + 64, p:p + 16]
                        ww = w[o:o + 8, kk:kk + 64]
                        part += self.mitchell_torch(
                            ww[:, :, None], aa[None, :, :]).sum(dim=1)
            return result

        torch.manual_seed(441)
        a = torch.randn(71, 19)
        w = torch.randn(11, 71)
        torch.testing.assert_close(mitchell_matrix(a, w), original_matrix(a, w),
                                   rtol=0, atol=0)

        layer = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=True)
        x = torch.randn(1, 2, 5, 6)
        patches = F.unfold(x, (3, 3), padding=1)[0]
        weights = layer.weight.reshape(4, -1)
        expected = original_matrix(patches, weights)
        expected += layer.bias[:, None]
        expected = expected.reshape(1, 4, 5, 6)
        torch.testing.assert_close(self.MitchellConv2d(layer)(x), expected,
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
