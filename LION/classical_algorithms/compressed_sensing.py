"""Compressed sensing utilities: wavelet transforms, composite operators, FISTA solver."""

from __future__ import annotations

import math
from tabnanny import verbose

import torch
from tqdm import tqdm

from LION.operators import Operator, PhotocurrentMapOp
from LION.utils.math import power_method_torch


class CompositeOp(Operator):
    """Composite linear operator A = Phi Psi^{-1} and its adjoint.

    Parameters
    ----------
    wavelet : Wavelet2D
        Wavelet transform object.
    phi : PhotocurrentMapOp
        Photocurrent mapping operator.
    device : str or torch.device, optional
        Device for computations. If None, uses wavelet.device.
    """

    def __init__(
        self,
        wavelet: Operator,
        phi: PhotocurrentMapOp,
        device: str | torch.device | None = None,
    ):
        self.wavelet = wavelet
        self.phi = phi
        self.device = torch.device(device)

    def __call__(self, w: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward projection.

        Parameters
        ----------
        w : torch.Tensor
            Wavelet coefficients, shape (Nw,).
        out : torch.Tensor | None, optional
            Optional output holder to store the output measurements.

        Returns
        -------
        torch.Tensor
            Predicted measurements, shape (M,).
        """
        return self.forward(w, out=out)

    def forward(self, w: torch.Tensor, out=None) -> torch.Tensor:
        """Apply A = Phi Psi^{-1}.

        Parameters
        ----------
        w : torch.Tensor
            Wavelet coefficients, shape (Nw,).

        Returns
        -------
        y : torch.Tensor
            Predicted measurements, shape (M,).
        """
        if not isinstance(w, torch.Tensor):
            raise TypeError(f"Input w must be a torch.Tensor, got {type(w)}")

        w = w.to(self.device)
        x = self.wavelet.inverse(w)  # (H, W) on self.device

        # If PhotocurrentMapOp expects different shape, reshaping is applied here
        y = self.phi.forward(x)
        return y

    def adjoint(self, r: torch.Tensor, out=None) -> torch.Tensor:
        """Apply A^T = Psi Phi^T.

        Parameters
        ----------
        r : torch.Tensor
            Residual in measurement space, shape (M,).

        Returns
        -------
        g : torch.Tensor
            Gradient in wavelet coefficient space, shape (Nw,).
        """
        r = r.to(self.device)
        x_back = self.phi.adjoint(r)  # expected shape (H, W)
        g = self.wavelet.forward(x_back)
        return g

    @property
    def domain_shape(self):
        # Opposite of range_shape of wavelet since wavelet is Psi^{-1}
        return self.wavelet.range_shape

    @property
    def range_shape(self):
        # Opposite of domain_shape of wavelet since wavelet is Psi^{-1}
        return self.wavelet.domain_shape


def soft_threshold(v: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft thresholding operator."""
    return torch.sign(v) * torch.clamp(torch.abs(v) - tau, min=0.0)


def fista_l1(
    op: Operator,
    y: torch.Tensor,
    lam: float,
    max_iter: int = 200,
    tol: float = 1e-4,
    L: float | None = None,
    verbose: bool = False,
    progress_bar: bool = False,
) -> torch.Tensor:
    """Solve min_w 0.5||A w - y||_2^2 + lam ||w||_1 by FISTA.

    Parameters
    ----------
    y : torch.Tensor
        Measurements, shape (M,).
    lam : float
        l1 regularisation parameter.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Relative stopping threshold on w.
    L : float or None
        Lipschitz constant of A^T A. If None, estimated by power iteration.
    verbose : bool
        If True, prints basic progress.

    Returns
    -------
    w : torch.Tensor
        Estimated wavelet coefficient vector, shape (Nw,).
    """
    y = y.detach()
    device = y.device

    # Dimension inferred from one adjoint call
    w0: torch.Tensor = op.adjoint(torch.zeros_like(y))
    n: int = w0.numel()

    if L is None:
        L = power_method_torch(op, device=device)
    step = 1.0 / (L + 1e-12)

    w = torch.zeros(n, dtype=torch.float32, device=device)
    z = w.clone()
    t = 1.0

    iterator = range(max_iter)
    if progress_bar:
        iterator = tqdm(iterator, desc="FISTA l1")
    for k in iterator:
        Az = op(z)
        grad = op.adjoint(Az - y)  # gradient of data term, shape (n,)

        w_next = soft_threshold(z - step * grad, lam * step)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = w_next + (t - 1.0) / t_next * (w_next - w)

        rel_change = torch.norm(w_next - w) / (torch.norm(w) + 1e-8)
        w = w_next
        t = t_next

        if verbose:
            data_term = 0.5 * torch.norm((op(w)) - y).pow(2).item()
            l1_term = lam * torch.norm(w, p=1).item()
            print(
                f"Iter {k:4d}  f = {data_term + l1_term:.4e}  "
                f"rel_change = {rel_change.item():.2e}  tol = {tol:.2e}  "
                f"rel_change < tol: {rel_change.item() < tol}"
            )

        if rel_change.item() < tol:
            break

    return w


class DebiasOp(Operator):
    """Debiasing least squares operator on the support of w.

    Parameters
    ----------
    operator : Operator
        Forward operator A.
    y : torch.Tensor
        Measurements, shape (M,).
    w : torch.Tensor
        l1 minimiser, shape (Nw,).
    support_tol : float
        Threshold defining nonzero support.
    """

    def __init__(
        self,
        op: Operator,
        y: torch.Tensor,
        w: torch.Tensor,
        support_tol: float = 1e-3,
    ):
        self.op = op
        self.y = y
        self.w = w
        self.support_tol = support_tol

        self.support = torch.nonzero(
            torch.abs(w) > support_tol, as_tuple=False
        ).squeeze(1)
        self.device = op.device

    def __call__(self, v: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward projection on the support."""
        return self.forward(v, out=out)

    def forward(self, v: torch.Tensor, out=None) -> torch.Tensor:
        """Apply A_s on the support."""
        w_full = torch.zeros_like(self.w)
        w_full[self.support] = v
        return self.op.forward(w_full, out=out)

    def adjoint(self, r: torch.Tensor, out=None) -> torch.Tensor:
        """Apply A_s^T on the support."""
        g_full = self.op.adjoint(r, out=out)
        return g_full[self.support]

    @property
    def domain_shape(self):
        return (self.support.numel(),)

    @property
    def range_shape(self):
        return self.op.range_shape


def debias_ls(
    op: Operator,
    y: torch.Tensor,
    w: torch.Tensor,
    support_tol: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-5,
    progress_bar: bool = False,
) -> torch.Tensor:
    """Debiasing least squares on the support of w.

    Parameters
    ----------
    y : torch.Tensor
        Measurements, shape (M,).
    w : torch.Tensor
        l1 minimiser, shape (Nw,).
    support_tol : float
        Threshold defining nonzero support.
    max_iter : int
        Maximum number of gradient descent iterations.
    tol : float
        Relative stopping threshold.

    Returns
    -------
    w_deb : torch.Tensor
        Debiased coefficient vector, shape (Nw,).
    """
    device = w.device

    support = torch.nonzero(torch.abs(w) > support_tol, as_tuple=False).squeeze(1)
    if support.numel() == 0:
        return w.clone()

    op_s = DebiasOp(op, y, w, support_tol=support_tol)

    L = power_method_torch(op_s, device=device)
    step = 1.0 / (L + 1e-12)

    v = w[support].clone()

    iterator = range(max_iter)
    if progress_bar:
        iterator = tqdm(iterator, desc="Debiasing LS")
    for _ in iterator:
        r = op_s(v) - y
        grad = op_s.adjoint(r)
        v_next = v - step * grad

        rel_change = torch.norm(v_next - v) / (torch.norm(v) + 1e-8)
        v = v_next

        if verbose:
            data_term = 0.5 * torch.norm(op_s(v) - y).pow(2).item()
            print(
                f"Debiasing LS  f = {data_term:.4e}  "
                f"rel_change = {rel_change.item():.2e}  tol = {tol:.2e}  "
                f"rel_change < tol: {rel_change.item() < tol}"
            )

        if rel_change.item() < tol:
            break

    w_deb = torch.zeros_like(w)
    w_deb[support] = v
    return w_deb
