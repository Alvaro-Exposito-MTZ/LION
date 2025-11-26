"""2D Haar wavelet transform with exact adjoint on GPU."""

import math

import torch
from spyrit.core.torch import fwht as sfwht

from LION.operators.Operator import Operator


def fwht(x: torch.Tensor, dim):
    """2D fast Walsh-Hadamard transform over the last two dimensions."""
    x = sfwht(x, dim=dim[-1])  # width
    x = sfwht(x, dim=dim[-2])  # height
    return x


def _haar2d_forward_level(x: torch.Tensor) -> torch.Tensor:
    """One-level 2D Haar analysis on the last two dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (..., H, W) with even H, W.

    Returns
    -------
    torch.Tensor
        Coefficients arranged as [LL; HL] vertically and [.. | ..] horizontally.
    """
    H, W = x.shape[-2:]
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"H and W must be even, got H={H}, W={W}")

    # Row transform (horizontal)
    lo = (x[..., :, 0::2] + x[..., :, 1::2]) / math.sqrt(2.0)
    hi = (x[..., :, 0::2] - x[..., :, 1::2]) / math.sqrt(2.0)
    tmp = torch.cat([lo, hi], dim=-1)  # (..., H, W)

    # Column transform (vertical)
    lo2 = (tmp[..., 0::2, :] + tmp[..., 1::2, :]) / math.sqrt(2.0)
    hi2 = (tmp[..., 0::2, :] - tmp[..., 1::2, :]) / math.sqrt(2.0)
    out = torch.cat([lo2, hi2], dim=-2)  # (..., H, W)

    return out


def _haar2d_inverse_level(c: torch.Tensor) -> torch.Tensor:
    """One-level 2D Haar synthesis (inverse of _haar2d_forward_level)."""
    H, W = c.shape[-2:]
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"H and W must be even, got H={H}, W={W}")

    H2 = H // 2
    W2 = W // 2

    # Subbands
    LL = c[..., :H2, :W2]
    HL = c[..., H2:, :W2]
    LH = c[..., :H2, W2:]
    HH = c[..., H2:, W2:]

    # Invert column transform for left and right halves separately
    tmp_left = torch.zeros_like(c[..., :H, :W2])
    tmp_left[..., 0::2, :] = (LL + HL) / math.sqrt(2.0)
    tmp_left[..., 1::2, :] = (LL - HL) / math.sqrt(2.0)

    tmp_right = torch.zeros_like(c[..., :H, W2:])
    tmp_right[..., 0::2, :] = (LH + HH) / math.sqrt(2.0)
    tmp_right[..., 1::2, :] = (LH - HH) / math.sqrt(2.0)

    tmp = torch.cat([tmp_left, tmp_right], dim=-1)  # (..., H, W)

    # Invert row transform
    W2 = W // 2
    lo = tmp[..., :, :W2]
    hi = tmp[..., :, W2:]
    out = torch.zeros_like(tmp)
    out[..., :, 0::2] = (lo + hi) / math.sqrt(2.0)
    out[..., :, 1::2] = (lo - hi) / math.sqrt(2.0)

    return out


def _haar2d_forward(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Multi-level 2D Haar analysis on the last two dimensions."""
    H, W = x.shape[-2:]
    out = x.clone()
    for level in range(levels):
        h = H // (2**level)
        w = W // (2**level)
        region = out[..., :h, :w]
        out[..., :h, :w] = _haar2d_forward_level(region)
    return out


def _haar2d_inverse(coeffs: torch.Tensor, levels: int) -> torch.Tensor:
    """Multi-level 2D Haar synthesis (inverse of _haar2d_forward)."""
    H, W = coeffs.shape[-2:]
    out = coeffs.clone()
    for level in reversed(range(levels)):
        h = H // (2**level)
        w = W // (2**level)
        region = out[..., :h, :w]
        out[..., :h, :w] = _haar2d_inverse_level(region)
    return out


class Wavelet2D_Haar(Operator):
    """2D Haar wavelet transform on GPU with exact adjoint.

    Parameters
    ----------
    shape : tuple of int
        Image shape (H, W), both powers of two.
    wavelet_name : str, optional
        Currently ignored; Haar (db1) is used.
    level : int or None
        Number of decomposition levels. If None, uses the maximum valid level.
    device : str or torch.device
        Device where tensors are placed.
    """

    def __init__(
        self,
        shape,
        wavelet_name: str = "haar",
        level: int | None = None,
        device: str | torch.device = "cpu",
    ):
        self.shape = tuple(shape)
        H, W = self.shape
        if (H & (H - 1)) != 0 or (W & (W - 1)) != 0:
            raise ValueError(f"H and W must be powers of two, got H={H}, W={W}")

        if level is None:
            self.level = int(math.floor(math.log2(min(H, W))))
        else:
            self.level = int(level)

        if self.level < 1:
            raise ValueError(f"Number of levels must be >= 1, got {self.level}")

        self.device = torch.device(device)
        self.size = H * W  # orthonormal transform, same dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wavelet analysis: image -> flat coefficient vector.

        Parameters
        ----------
        x : torch.Tensor
            Real image, shape (H, W).

        Returns
        -------
        torch.Tensor
            Flattened Haar coefficients, shape (H * W,).
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected x with 2 dims (H, W), got shape {tuple(x.shape)}"
            )

        H, W = self.shape
        if x.shape[-2:] != (H, W):
            raise ValueError(f"Expected x shape {(H, W)}, got {tuple(x.shape[-2:])}")

        x4 = x.to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        coeffs = _haar2d_forward(x4, self.level)  # (1, 1, H, W)
        return coeffs.view(-1)

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Wavelet synthesis: flat coefficient vector -> image."""
        H, W = self.shape
        if w.numel() != H * W:
            raise ValueError(f"Expected w with {H * W} elements, got {w.numel()}")

        x4 = w.to(self.device).view(1, 1, H, W)
        img4 = _haar2d_inverse(x4, self.level)  # (1, 1, H, W)
        return img4[0, 0]

    def to(self, device: str | torch.device):
        """Move internal tensors to a given device."""
        self.device = torch.device(device)
        return self
