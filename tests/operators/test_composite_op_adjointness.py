import torch
from LION.classical_algorithms.compressed_sensing import CompositeOp
from LION.operators import PhotocurrentMapOp, Subsampler, Wavelet2D_DB4
from tests.helper import dotproduct_adjointness_test


def test_composite_op_adjointness():
    """Test adjoint property of A = Phi Psi^{-1}."""
    J = 9  # 512 x 512 images
    N = 1 << J
    subtract_from_J = 1
    coarseJ = J - subtract_from_J
    delta = 1.0 / 4
    image_shape = (N, N)

    # Wavelet transform Psi
    wavelet = Wavelet2D_DB4(image_shape, wavelet_name="db4")

    # Photocurrent mapping operator Phi
    subsampler = Subsampler(n=N * N, coarseJ=coarseJ, delta=delta)
    phi = PhotocurrentMapOp(J=J, subsampler=subsampler)

    # Composite operator A = Phi Psi^{-1}
    operator = CompositeOp(wavelet, phi, device=torch.get_default_device())

    # Domain: wavelet coefficients (length = number of coefficients)
    n_w = wavelet.size
    x = torch.rand(n_w)

    # Codomain: measurements (infer length by one forward pass)
    y0 = operator(x * 0.0)  # same shape as A(x)
    y = torch.rand_like(y0)

    dotproduct_adjointness_test(operator, x, y)
