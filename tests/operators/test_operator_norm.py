"""Tests for computing the operator norm of operators."""

import numpy as np
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.operators import (
    PhotocurrentMapOp,
    PhotocurrentMapOpNumpy,
    Subsampler,
    WalshHadamardTransform2D,
    WalshHadamardTransform2DNumpy,
    Wavelet2D,
    Wavelet2DNumpy,
)
from LION.utils.math import power_method_numpy, power_method_torch


def test_ct_operator_norm_torch():
    """Test with CT operator using default geometry."""

    geometry = Geometry.default_parameters()
    ct_op = make_operator(geometry)

    ct_op_norm = power_method_torch(ct_op)
    ct_op_norm_true = 4.0  # known from literature for this setup

    torch.testing.assert_close(ct_op_norm.item(), ct_op_norm_true, atol=1e-2, rtol=1e-2)


def test_ct_operator_norm_numpy():
    """Test with CT operator for Numpy arrays using default geometry."""

    geometry = Geometry.default_parameters()
    ct_op = make_operator(geometry)

    ct_op_norm = power_method_numpy(ct_op)
    ct_op_norm_true = 4.0  # known from literature for this setup

    np.testing.assert_allclose(ct_op_norm.item(), ct_op_norm_true, atol=1e-2, rtol=1e-2)


def test_pcm_operator_norm_torch():
    """Test with photocurrent mapping operator with undersampling."""

    J = 4
    H = W = 1 << J  # 16x16 image
    delta = 1.0 / 4
    coarseJ = J - 1
    subsampler = Subsampler(n=H * W, delta=delta, coarseJ=coarseJ)
    pcm_op = PhotocurrentMapOp(J=J, subsampler=subsampler)

    pcm_op_norm = power_method_torch(pcm_op)
    pcm_op_norm_true = 2.0  # Ref: "..." paper

    torch.testing.assert_close(
        pcm_op_norm.item(), pcm_op_norm_true, atol=1e-2, rtol=1e-2
    )


def test_pcm_operator_norm_numpy():
    """Test with photocurrent mapping operator for Numpy arrays with undersampling."""

    J = 4
    H = W = 1 << J  # 16x16 image
    delta = 1.0 / 4
    coarseJ = J - 1
    subsampler = Subsampler(n=H * W, delta=delta, coarseJ=coarseJ)
    pcm_op = PhotocurrentMapOpNumpy(J=J, subsampler=subsampler)

    pcm_op_norm = power_method_numpy(pcm_op)
    pcm_op_norm_true = 2.0  # Ref: "..." paper

    np.testing.assert_allclose(
        pcm_op_norm.item(), pcm_op_norm_true, atol=1e-2, rtol=1e-2
    )


def test_wht_operator_norm_torch():
    """Test with Walsh-Hadamard Transform operator."""

    J = 4
    H = W = 1 << J  # 16x16 image
    wht_op = WalshHadamardTransform2D(height=H, width=W)

    wht_op_norm = power_method_torch(wht_op)
    rng = torch.Generator().manual_seed(0)
    x = torch.rand((H, W), dtype=torch.float32, generator=rng)
    y = wht_op(x)
    ratio = torch.norm(y) / torch.norm(x)
    # Since our operator is not normalized,  ||WHT x|| / ||x|| = ||WHT||.
    torch.testing.assert_close(ratio.item(), wht_op_norm.item(), atol=1e-4, rtol=1e-4)

    # WHT has norm sqrt(height*width) = sqrt(N*N) = N
    wht_op_norm_true = float(N)
    torch.testing.assert_close(
        wht_op_norm.item(), wht_op_norm_true, atol=1e-2, rtol=1e-2
    )


def test_wht_operator_norm_numpy():
    """Test with Walsh-Hadamard Transform operator for NumPy arrays."""

    J = 4
    H = W = 1 << J  # 16x16 image
    wht_op = WalshHadamardTransform2DNumpy(height=H, width=W)

    wht_op_norm = power_method_numpy(wht_op)
    rng = np.random.default_rng(seed=0)
    x = rng.random((H, W)).astype(np.float32)
    y = wht_op(x)
    ratio = np.linalg.norm(y) / np.linalg.norm(x)
    # Since our operator is not normalized,  ||WHT x|| / ||x|| = ||WHT||.
    np.testing.assert_allclose(ratio, wht_op_norm, atol=1e-4, rtol=1e-4)

    # WHT has norm sqrt(height*width) = sqrt(N*N) = N
    wht_op_norm_true = float(N)
    np.testing.assert_allclose(
        wht_op_norm.item(), wht_op_norm_true, atol=1e-2, rtol=1e-2
    )


def test_wavelet_operator_norm_torch():
    """Test with Daubechies 4 wavelet transform operator."""

    image_shape = (16, 16)
    wavelet_op = Wavelet2D(image_shape, wavelet_name="db4")

    wavelet_op_norm = power_method_torch(wavelet_op)
    wavelet_op_norm_true = 1.0  # orthonormal wavelet

    torch.testing.assert_close(
        wavelet_op_norm.item(), wavelet_op_norm_true, atol=1e-4, rtol=1e-4
    )


def test_wavelet_operator_norm_numpy():
    """Test with Daubechies 4 wavelet transform operator for Numpy arrays."""

    image_shape = (16, 16)
    wavelet_op_np = Wavelet2DNumpy(image_shape, wavelet_name="db4")

    wavelet_op_norm = power_method_numpy(wavelet_op_np)
    wavelet_op_norm_true = 1.0  # orthonormal wavelet

    np.testing.assert_allclose(
        wavelet_op_norm.item(), wavelet_op_norm_true, atol=1e-4, rtol=1e-4
    )
