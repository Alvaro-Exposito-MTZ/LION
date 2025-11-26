"""Thin wrapper around tomosipo's Operator class for linear tomographic projection."""

from __future__ import annotations

import numpy as np
import tomosipo as ts
import torch

from LION.operators.Operator import Operator


class TomographicProjOp(Operator):
    def __init__(self, ts_operator: ts.Operator.Operator, device=None):
        super().__init__(device=device)
        self._ts = ts_operator

    def __call__(
        self,
        x: torch.Tensor | np.ndarray | ts.Data.Data,
        out: torch.Tensor | np.ndarray | ts.Data.Data | None = None,
    ) -> torch.Tensor | np.ndarray | ts.Data.Data:
        """Apply the forward projection.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray | ts.Data.Data
            The input volume dataset to which the forward projection is applied.
        out : torch.Tensor | np.ndarray | ts.Data.Data | None, optional
            Optional output holder to store the output projection dataset.

        Returns
        -------
        torch.Tensor | np.ndarray | ts.Data.Data
            The projection dataset on which the volume has been forward
            projected.
        """
        return self.forward(x, out=out)

    def forward(
        self,
        x: torch.Tensor | np.ndarray | ts.Data.Data,
        out: torch.Tensor | np.ndarray | ts.Data.Data | None = None,
    ) -> torch.Tensor | np.ndarray | ts.Data.Data:
        """Apply the forward of TomographicProjOp.

        .. note::
            Prefer calling the instance of the TomographicProjOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._ts._fp(x, out=out)

    def adjoint(
        self,
        y: torch.Tensor | np.ndarray | ts.Data.Data,
        out: torch.Tensor | np.ndarray | ts.Data.Data | None = None,
    ) -> torch.Tensor | np.ndarray | ts.Data.Data:
        """Apply the backprojection.

        Parameters
        ----------
        y : torch.Tensor | np.ndarray | ts.Data.Data
            The input projection dataset to which the backprojection is applied.
        out : torch.Tensor | np.ndarray | ts.Data.Data | None, optional
            Optional output holder to store the output volume dataset.

        Returns
        -------
        torch.Tensor | np.ndarray | ts.Data.Data
            The volume dataset on which the projection dataset has been
            backprojected.
        """
        return self._ts._bp(y, out=out)

    def __getattr__(self, name):
        return getattr(self._ts, name)

    @property
    def domain_shape(self):
        return self._ts.domain_shape

    @property
    def range_shape(self):
        return self._ts.range_shape
