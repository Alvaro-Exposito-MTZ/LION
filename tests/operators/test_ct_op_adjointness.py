"""Tests for the adjointness property of the CT operator."""

import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from tests.helper import dotproduct_adjointness_test


def test_ct_op_adjointness():
    """Test CT operator adjoint property."""
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    # Check the default operator shapes
    assert operator.domain_shape == (1, 512, 512)
    assert operator.range_shape == (1, 360, 900)

    # Create a test input for the forward and backward projections
    test_volume = torch.rand(*operator.domain_shape)
    test_projection = torch.rand(*operator.range_shape)
    dotproduct_adjointness_test(operator, test_volume, test_projection)
