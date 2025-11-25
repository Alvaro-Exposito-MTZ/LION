"""Tests for CT operator wrapped with tomosipo's to_autograd function."""

import pytest
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from tomosipo.torch_support import to_autograd


def test_ct_autograd_op_forward_and_backward():
    """Check that to_autograd wraps the CT operator correctly and supports backprop."""
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)
    autograd_operator = to_autograd(operator)

    torch.manual_seed(0)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    output_tensor = autograd_operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad
    assert input_tensor.grad is None

    output_tensor.mean().backward()

    assert input_tensor.grad is not None
    assert isinstance(input_tensor.grad, torch.Tensor)
    assert input_tensor.grad.shape == input_tensor.shape
    assert torch.isfinite(input_tensor.grad).all()


def test_ct_autograd_op_matches_original_operator():
    """Check that the autograd wrapper produces the same output as the original operator."""
    import numpy as np

    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)
    autograd_operator = to_autograd(operator)

    torch.manual_seed(1)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    input_np = input_tensor.detach().cpu().numpy()
    output_np = operator(input_np)
    output_autograd = autograd_operator(input_tensor).detach().cpu().numpy()

    np.testing.assert_allclose(output_autograd, output_np, rtol=1e-5, atol=1e-5)


def test_original_op_does_not_propagate_grad():
    """Check that the original operator output does not require gradients and backward fails."""
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    torch.manual_seed(2)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    # Original operator is not autograd aware
    output_tensor = operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad is False

    # Backprop should fail with the expected autograd error
    with pytest.raises(
        RuntimeError, match="does not require grad and does not have a grad_fn"
    ):
        output_tensor.mean().backward()
