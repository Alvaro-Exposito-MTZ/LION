import torch
from LION.operators import Wavelet2D_DB4
from tests.helper import dotproduct_adjointness_test


def test_wavelet_db4_adjointness():
    J = 9
    N = 1 << J
    image_shape = (N, N)

    operator = Wavelet2D_DB4(image_shape, wavelet_name="db4")

    x = torch.rand(*image_shape)

    # infer coefficient shape
    c0 = operator(x * 0.0)
    w = torch.rand_like(c0)

    dotproduct_adjointness_test(operator, x, w)
