"""Tests for the adjointness property of the photocurrent mapping operator."""

import torch
from LION.operators import PhotocurrentMapOp, Subsampler
from tests.helper import dotproduct_adjointness_test


def test_pcm_op_adjointness():
    """Test photocurrent mapping operator adjoint property."""
    J = 9  # 512x512 images
    N = 1 << J
    subsampler = Subsampler(n=N * N, delta=0.25, coarseJ=J - 1)
    operator = PhotocurrentMapOp(J=J, subsampler=subsampler)

    # Check the default operator shapes
    assert operator.domain_shape == (512, 512)
    assert operator.range_shape == (65536,)

    # Create a test input for the forward and backward projections
    test_image = torch.rand(*operator.domain_shape)
    test_measurement = torch.rand(*operator.range_shape)
    dotproduct_adjointness_test(operator, test_image, test_measurement)

    # Test pseudoinverse runs without error
    assert operator.pseudo_inv(test_measurement) is not None
