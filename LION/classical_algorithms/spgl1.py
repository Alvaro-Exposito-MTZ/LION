import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from spgl1 import spgl1

from LION.operators.Operator import Operator


def spgl1_torch(
    op: Operator,
    y: torch.Tensor,
    *,  # All arguments below are keyword-only
    lam: float,
    iter_lim: int = 200,
    mode: str = "lasso",
    **spgl1_kwargs,
) -> torch.Tensor:
    r"""Solve an l1 sparse reconstruction using SPGL1, wrapping torch operators.

    This is a thin wrapper around the Python SPGL1 solver ``spgl1.spgl1`` that
    uses torch operators for matrix-vector products. SPGL1 is a spectral
    projected-gradient method for constrained l1 problems; see
    [BergFriedlander2008]_ and [BergFriedlander2010]_.

    This wrapper is built on top of the Python implementation ``spgl1.spgl1`` and
    uses the same calling convention (argument names and behaviour); see
    [SPGL1Python]_ for details.

    It solves either

    - Lasso (basis pursuit along the Pareto curve; see [BergFriedlander2008]_):
      :math:`\\min_w \\tfrac12\\lVert A w - y \\rVert_2^2`
      subject to :math:`\\lVert w \\rVert_1 \\le \\tau`
      when ``mode == 'lasso'`` with :math:`\\tau = \\lambda`.

    - Basis pursuit denoise (BPDN; see [BergFriedlander2010]_):
      :math:`\\min_w \\lVert w \\rVert_1`
      subject to :math:`\\lVert A w - y \\rVert_2 \\le \\sigma`
      when ``mode == 'bpdn'`` with :math:`\\sigma = \\lambda`.

    Parameters
    ----------
    op : Operator
        Linear operator implementing the forward map and its adjoint. It is
        called as ``op(w)`` and ``op.adjoint(r)``.
    y : torch.Tensor
        Measurements, shape ``(M,)``.
    lam : float
        Hyperparameter. Interpreted as:
        - ``mode == 'lasso'``: ``lam = tau`` is the l1 budget.
        - ``mode == 'bpdn'``: ``lam = sigma`` is the residual bound.
    mode : {'lasso', 'bpdn'}
        Which SPGL1 formulation to use; see [BergFriedlander2008]_ and
        [BergFriedlander2010]_ for the underlying optimisation problems.
    spgl1_kwargs : dict
        Extra keyword args forwarded to ``spgl1.spgl1`` (for example
        tolerances or iteration limits; see [SPGL1Python]_).

    Returns
    -------
    w_hat : torch.Tensor
        Estimated coefficient vector in the same shape as ``op.adjoint(y*0)``.

    References
    ----------
    .. [BergFriedlander2008] E. van den Berg and M. P. Friedlander, "Probing
       the Pareto frontier for basis pursuit solutions", SIAM Journal on
       Scientific Computing, 31(2):890-912, 2008.
    .. [BergFriedlander2010] E. van den Berg and M. P. Friedlander, "Sparse
       optimisation with least-squares constraints", TR-2010-02, Department of
       Computer Science, University of British Columbia, 2010.
    .. [SPGL1Python] SPGL1: Spectral Projected Gradient for L1 minimisation,
       Python package documentation, https://spgl1.readthedocs.io/
    """
    device = y.device
    y = y.detach()

    # Infer coefficient shape from one adjoint call
    with torch.no_grad():
        w0 = op.adjoint(torch.zeros_like(y))
    w0 = w0.detach()
    n_w = w0.numel()
    n_y = y.numel()

    if mode == "lasso":
        tau = float(lam)
        sigma = 0.0
    elif mode == "bpdn":
        tau = 0.0
        sigma = float(lam)
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'lasso' or 'bpdn'.")

    def matvec(w_np: np.ndarray) -> np.ndarray:
        w_t = torch.from_numpy(w_np.astype(np.float32)).to(device).view_as(w0)
        y_t = op(w_t)
        return y_t.detach().cpu().numpy().ravel()

    def rmatvec(r_np: np.ndarray) -> np.ndarray:
        r_t = torch.from_numpy(r_np.astype(np.float32)).to(device).view_as(y)
        g_t = op.adjoint(r_t)
        return g_t.detach().cpu().numpy().ravel()

    A_linop = LinearOperator(
        shape=(n_y, n_w),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float32,
    )

    y_np = y.detach().cpu().numpy().ravel()
    x0_np = np.zeros(n_w, dtype=np.float32)

    x_np, _, _, _ = spgl1(
        A_linop,
        y_np,
        tau=tau,
        sigma=sigma,
        x0=x0_np,
        iter_lim=iter_lim,
        **spgl1_kwargs,
    )

    w_hat = torch.from_numpy(x_np.astype(np.float32)).to(device).view_as(w0)
    return w_hat
