"""Walsh-Hadamard Transform 2D Operator."""

from __future__ import annotations

import numpy as np
import torch
from spyrit.core.torch import fwht

from LION.operators.Operator import Operator


class WalshHadamardTransform2D(Operator):
    """Walsh-Hadamard Transform operator.

    Parameters
    ----------
    height : int
        Height of the 2D input vector. Must be a power of two.
    width : int
        Width of the 2D input vector. Must be a power of two.
    device : str or torch.device
        Device where tensors are placed.
    """

    def __init__(
        self, height: int, width: int, device: str | torch.device | None = None
    ):
        """Walsh-Hadamard Transform operator.

        Parameters
        ----------
        height : int
            Height of the 2D input vector. Must be a power of two.
        width : int
            Width of the 2D input vector. Must be a power of two.
        device : str or torch.device
            Device where tensors are placed.
        """
        super().__init__(device=device)
        if (height & (height - 1)) != 0:
            raise ValueError("height must be a power of two.")
        if (width & (width - 1)) != 0:
            raise ValueError("width must be a power of two.")
        self.height = height
        self.width = width
        self.N = height * width

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the Walsh-Hadamard Transform.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of shape (height, width).

        Returns
        -------
        torch.Tensor
            Transformed vector of shape (N,).
        """
        return self.forward(x, out=out)

    def forward(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the Walsh-Hadamard Transform.

        .. note::
            Prefer calling the instance of the WalshHadamardTransform2D operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if len(x.shape) != 2 or x.shape != (self.height, self.width):
            raise ValueError(
                f"Expected input of shape ({self.height}, {self.width}), got {x.shape}"
            )
        return fwht(x.to(self.device).ravel(), order=False)

    def adjoint(self, y: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the adjoint of the Walsh-Hadamard Transform.

        Parameters
        ----------
        y : torch.Tensor
            Input vector of shape (N,).

        Returns
        -------
        torch.Tensor
            Adjoint transformed vector of shape (height, width).
        """
        if len(y.shape) != 1 or y.shape[0] != self.N:
            raise ValueError(f"Expected input of shape ({self.N},), got {y.shape}")

        return fwht(y.to(self.device), order=False).view(self.height, self.width)

    def inverse(self, y: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the pseudo-inverse of the Walsh-Hadamard Transform.

        Parameters
        ----------
        y : torch.Tensor
            Input vector of shape (N,).

        Returns
        -------
        torch.Tensor
            Pseudo-inverse transformed vector of shape (N,).
        """
        return self.adjoint(y) / self.N

    @property
    def domain_shape(self):
        return (self.height, self.width)

    @property
    def range_shape(self):
        return (self.N,)


class WalshHadamardTransform2DNumpy(Operator):
    """Walsh-Hadamard Transform operator for NumPy arrays.

    ::note::
        We prefer using the PyTorch version :class:`WalshHadamardTransform2D` unless there is a specific reason
        to use NumPy.

    Parameters
    ----------
    height : int
        Height of the 2D input vector. Must be a power of two.
    width : int
        Width of the 2D input vector. Must be a power of two.
    """

    def __init__(self, height: int, width: int):
        """Walsh-Hadamard Transform operator.

        Parameters
        ----------
        height : int
            Height of the 2D input vector. Must be a power of two.
        width : int
            Width of the 2D input vector. Must be a power of two.
        """
        super().__init__(device=None)
        if (height & (height - 1)) != 0:
            raise ValueError("height must be a power of two.")
        if (width & (width - 1)) != 0:
            raise ValueError("width must be a power of two.")
        self.height = height
        self.width = width
        self.N = height * width

    def __call__(self, x: np.ndarray, out=None) -> np.ndarray:
        """Apply the Walsh-Hadamard Transform.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (height, width).

        Returns
        -------
        np.ndarray
            Transformed vector of shape (N,).
        """
        return self.forward(x, out=out)

    def forward(self, x: np.ndarray, out=None) -> np.ndarray:
        """Apply the Walsh-Hadamard Transform.

        .. note::
            Prefer calling the instance of the WalshHadamardTransform2D operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if len(x.shape) != 2 or x.shape != (self.height, self.width):
            raise ValueError(
                f"Expected input of shape ({self.height}, {self.width}), got {x.shape}"
            )
        x = torch.from_numpy(x.astype(np.float32))
        return fwht(x.ravel(), order=False).numpy()

    def adjoint(self, y: np.ndarray, out=None) -> np.ndarray:
        """Apply the adjoint of the Walsh-Hadamard Transform.

        Parameters
        ----------
        y : np.ndarray
            Input vector of shape (N,).

        Returns
        -------
        np.ndarray
            Adjoint transformed vector of shape (height, width).
        """
        if len(y.shape) != 1 or y.shape[0] != self.N:
            raise ValueError(f"Expected input of shape ({self.N},), got {y.shape}")

        y = torch.from_numpy(y.astype(np.float32))
        return fwht(y, order=False).view(self.height, self.width).numpy()

    def inverse(self, y: np.ndarray, out=None) -> np.ndarray:
        """Apply the pseudo-inverse of the Walsh-Hadamard Transform.

        Parameters
        ----------
        y : np.ndarray
            Input vector of shape (N,).

        Returns
        -------
        np.ndarray
            Pseudo-inverse transformed vector of shape (N,).
        """
        return self.adjoint(y) / self.N

    @property
    def domain_shape(self):
        return (self.height, self.width)

    @property
    def range_shape(self):
        return (self.N,)
