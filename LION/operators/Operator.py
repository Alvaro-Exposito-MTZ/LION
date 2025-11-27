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

    The main purpose of this class is to define a common interface for all
    operators in LION. The most important API for algorithms includes
    ``forward``, ``adjoint``, and ``domain_shape`` / ``range_shape``.

    This class should be subclassed to implement specific operators.

    Parameters
    ----------
    device : torch.device | str | None, optional
        Device for computations. If None, computations are done on the
        default device.
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

        .. note::
            Usually this is just a wrapper for the ``forward`` method:
            ``return self.forward(x, out=out)``.
            We want the user to write this explicitly in their subclasses
            so that they can add the docstring in the ``__call__`` method
            instead of the ``forward`` method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.

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
        raise NotImplementedError(
            "__call__ must be implemented by subclasses with docstring. "
            "Usually this is just a wrapper for the forward method: "
            "return self.forward(x, out=out)"
        )

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

    def gram(
        self,
        x: torch.Tensor | np.ndarray | ts.Data.Data,
        out: torch.Tensor | np.ndarray | ts.Data.Data | None = None,
    ) -> torch.Tensor | np.ndarray | ts.Data.Data:
        """Apply the Gram operator.

        For a LinearOperator :math:`A`, the self-adjoint Gram operator is defined as :math:`A^H A`.

        .. note::
           This is the inherited default implementation.
        """
        return self.adjoint(self.forward(x, out=None), out=out)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the inverse of the operator applied to input x if exists.

        This method should only be implemented by subclasses that have a well-defined inverse.

        Parameters
        ----------
        x : torch.Tensor
            The input data to which the inverse operator is applied.

        Returns
        -------
        torch.Tensor
            The result of applying the inverse operator to x.
        """
        raise NotImplementedError(
            "The inverse method is not implemented, possibly because "
            "the operator does not have a well-defined inverse."
        )

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
