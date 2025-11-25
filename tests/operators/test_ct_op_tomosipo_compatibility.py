"""Tests for backward compatibility with tomosipo Operator."""

import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator


def test_ct_op_backward_compatibility_with_tomosipo():
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    # Check the default operator shapes
    assert operator.domain_shape == (1, 512, 512)
    assert operator.range_shape == (1, 360, 900)

    # Create a test input for the forward and backward projections
    test_volume = torch.rand(*operator.domain_shape)
    test_projection = torch.rand(*operator.range_shape)

    # Make sure that tomosipo Operator's attributes are accessible
    assert operator._fp(volume=test_volume, out=None) is not None
    assert operator._bp(projection=test_projection, out=None) is not None
    assert operator.transpose() is not None
    assert operator.T is not None
    assert operator.domain is not None
    assert operator.range is not None
