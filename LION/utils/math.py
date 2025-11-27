"""Utility math functions."""

from __future__ import annotations

import numpy as np
import torch

from LION.operators import Operator


def power_method(
    op: Operator, maxiter: int = 100, tol: float = 1e-6, uses_torch: bool = True
) -> float:
    """Estimate operator norm by power iteration.

    This is a wrapper function to choose between numpy and torch implementations.

    Parameters
    ----------
    op : Operator
        The operator for which to estimate the norm.
    maxiter : int
        Number of power iterations.
    tol : float
        Absolute tolerance for convergence. (TODO: consider adding relative tolerance?)
    uses_torch : bool
        Whether to use the torch implementation. We use torch by default.

    Returns
    -------
    sigma : float
        Estimated operator norm.
    """
    if uses_torch:
        return power_method_torch(op, maxiter=maxiter).item()
    return power_method_numpy(op, maxiter=maxiter, tol=tol).item()


def power_method_numpy(
    op: Operator, maxiter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """Estimate operator norm by power iteration using numpy.

    Parameters
    ----------
    op : Operator
        The operator for which to estimate the norm.
    maxiter : int
        Number of power iterations.
    tol : float
        Absolute tolerance for convergence. (TODO: consider adding relative tolerance?)

    Returns
    -------
    sigma : np.ndarray
        Estimated operator norm.
    """
    arr_old = np.random.rand(*op.domain_shape).astype(np.float32)
    error = tol + 1
    i = 0
    while error >= tol:

        # very verbose and inefficient for now
        omega = op(arr_old)
        alpha = np.linalg.norm(omega)
        u = (1.0 / alpha) * omega
        z = op.adjoint(u)
        beta = np.linalg.norm(z)
        arr = (1.0 / beta) * z
        error = np.linalg.norm(op(arr) - beta * u)
        sigma = beta
        arr_old = arr
        i += 1
        if i >= maxiter:
            return sigma

    return sigma


def power_method_torch(
    op: Operator,
    maxiter: int = 100,
    tol: float = 1e-6,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Estimate operator norm by power iteration using torch.

    Parameters
    ----------
    op : Operator
        The operator for which to estimate the norm.
    maxiter : int
        Number of power iterations.
    tol : float
        Absolute tolerance for convergence. (TODO: consider adding relative tolerance?)

    Returns
    -------
    sigma : torch.Tensor
        Estimated operator norm.
    """
    arr_old = torch.randn(*op.domain_shape, device=device)
    error = tol + 1
    i = 0
    while error >= tol:

        # very verbose and inefficient for now
        omega = op(arr_old)
        alpha = torch.linalg.norm(omega)
        u = (1.0 / alpha) * omega
        z = op.adjoint(u)
        beta = torch.linalg.norm(z)
        arr = (1.0 / beta) * z
        error = torch.linalg.norm(op(arr) - beta * u)
        sigma = beta
        arr_old = arr
        i += 1
        if i >= maxiter:
            return sigma

    return sigma


def test_convexity(net, x, device):
    # check convexity of the net numerically
    print("running a numerical convexity test...")
    n_trials = 100
    convexity = 0
    for trial in np.arange(n_trials):
        x1 = torch.rand(x.size()).to(device)
        x2 = torch.rand(x.size()).to(device)
        alpha = torch.rand(1).to(device)

        cvx_combo_of_input = net(alpha * x1 + (1 - alpha) * x2)
        cvx_combo_of_output = alpha * net(x1) + (1 - alpha) * net(x2)

        convexity += cvx_combo_of_input.mean() <= cvx_combo_of_output.mean()
    if convexity == n_trials:
        flag = True
        print("Passed convexity test!")
    else:
        flag = False
        print("Failed convexity test!")
    return flag
