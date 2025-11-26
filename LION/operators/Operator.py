"""Base class for operators in LION."""

from __future__ import annotations

import numpy as np
import tomosipo as ts
import torch


class Operator:
    """
    Base class for operators in LION.

    Operators represent mathematical operations that can be applied to data,
    such as forward and backward projections in imaging.

    This class should be subclassed to implement specific operators.
    """

    def __init__(self, device: torch.device | str | None = None):
        """Initialize the Operator."""
        self.device = device

    def __call__(
        self,
        x: torch.Tensor | np.ndarray | ts.Data.Data,
        out: torch.Tensor | np.ndarray | ts.Data.Data | None = None,
    ) -> torch.Tensor | np.ndarray | ts.Data.Data:
        """
        Apply the forward operation of the operator.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray | ts.Data
            Input to which the operator is applied.
        out : torch.Tensor | np.ndarray | ts.Data | None, optional
            Optional output holder to store the result.

        Returns
        -------
        torch.Tensor | np.ndarray | ts.Data
            Result of applying the forward operation.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def forward(
        self,
        x: torch.Tensor | np.ndarray | ts.Data.Data,
        out: torch.Tensor | np.ndarray | ts.Data.Data | None = None,
    ) -> torch.Tensor | np.ndarray | ts.Data.Data:
        """
        Apply the forward operation of Operator.

        .. note::
            Prefer calling the instance of the Operator operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def adjoint(
        self,
        y: torch.Tensor | np.ndarray | ts.Data.Data,
        out: torch.Tensor | np.ndarray | ts.Data.Data | None = None,
    ) -> torch.Tensor | np.ndarray | ts.Data.Data:
        """
        Apply the adjoint (backward) operation of the operator.

        Parameters
        ----------
        y : torch.Tensor | np.ndarray | ts.Data
            Input to which the adjoint operator is applied.
        out : torch.Tensor | np.ndarray | ts.Data | None, optional
            Optional output holder to store the result.

        Returns
        -------
        torch.Tensor | np.ndarray | ts.Data
            Result of applying the adjoint operation.
        """
        raise NotImplementedError("Adjoint method must be implemented by subclasses.")

    @property
    def domain_shape(self):
        """
        Get the shape of the image domain.

        Returns
        -------
            Shape of the image domain.
        """
        raise NotImplementedError("property must be implemented by subclasses.")

    @property
    def range_shape(self):
        """
        Get the shape of the data (measurement) domain.

        Returns
        -------
            Shape of the data (measurement) domain.
        """
        raise NotImplementedError("property must be implemented by subclasses.")
