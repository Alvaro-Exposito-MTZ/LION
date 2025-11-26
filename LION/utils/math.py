"""Utility math functions."""

from __future__ import annotations

import numpy as np
import torch

from LION.operators import Operator


def power_method(op: Operator, maxiter: int = 100, tol: float = 1e-6):
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
    maxiter: int = 20,
    device: str | torch.device | None = None,
) -> float:
    """Estimate operator norm by power iteration.

    Parameters
    ----------
    maxiter : int
        Number of power iterations.
    device : str or torch.device, optional
        Device for the power iteration.

    Returns
    -------
    L : float
        Approximate Lipschitz constant of grad f = A^T A w.
    """
    v = torch.randn(op.domain_shape, dtype=torch.float32, device=device)
    v = v / (torch.norm(v) + 1e-12)

    norm_AtAv = 0.0
    for _ in range(maxiter):
        Av = op(v)
        AtAv = op.adjoint(Av)
        norm_AtAv = torch.norm(AtAv).item()
        if norm_AtAv == 0.0:
            break
        v = AtAv / (norm_AtAv + 1e-12)

    return norm_AtAv


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
