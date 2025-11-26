import torch
from LION.operators import Operator, Wavelet2D_DB4, Wavelet2D_Haar
from tests.helper import dotproduct_adjointness_test


class WaveletAsOperator(Operator):
    """Treat Wavelet2D as a linear operator W with W.H."""

    def __init__(self, wavelet: Operator):
        self.wavelet = wavelet

    def __call__(self, x: torch.Tensor, out=None):
        return self.forward(x, out=out)

    def forward(self, x: torch.Tensor, out=None):
        # x is an image (H, W)
        return self.wavelet.forward(x)

    def adjoint(self, w: torch.Tensor, out=None):
        # w is coefficient vector
        return self.wavelet.inverse(w)


def test_wavelet_db4_adjointness():
    J = 9
    N = 1 << J
    image_shape = (N, N)

    wavelet = Wavelet2D_DB4(image_shape, wavelet_name="db4")
    operator = WaveletAsOperator(wavelet)

    x = torch.rand(*image_shape)

    # infer coefficient shape
    c0 = operator(x * 0.0)
    w = torch.rand_like(c0)

    dotproduct_adjointness_test(operator, x, w)


def test_wavelet_haar_adjointness():
    J = 9
    N = 1 << J
    image_shape = (N, N)

    wavelet = Wavelet2D_Haar(image_shape, wavelet_name="haar")
    operator = WaveletAsOperator(wavelet)

    x = torch.rand(*image_shape)

    # infer coefficient shape
    c0 = operator(x * 0.0)
    w = torch.rand_like(c0)

    dotproduct_adjointness_test(operator, x, w)
