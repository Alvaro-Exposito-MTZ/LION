import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from spgl1 import spgl1

from LION.operators.Operator import Operator


def spgl1_l1(
    op: Operator,
    y: torch.Tensor,
    lam: float,
    max_iter: int = 200,
    mode: str = "lasso",
    verbose: bool = False,
    spgl1_kwargs: dict | None = None,
) -> torch.Tensor:
    """Solve an l1 sparse reconstruction using SPGL1, wrapping torch operators.

    Parameters
    ----------
    y : torch.Tensor
        Measurements, shape (M,).
    lam : float
        Hyperparameter. Interpreted as:
        - mode == 'lasso': lam = tau = l1 budget.
        - mode == 'bpdn' : lam = sigma = residual bound.
    max_iter : int
        Maximum number of SPGL1 iterations (passed via kwargs if supported).
    mode : {'lasso', 'bpdn'}
        Which SPGL1 formulation to use.
    verbose : bool
        If True, increase SPGL1 verbosity (if supported).
    spgl1_kwargs : dict or None
        Extra keyword args forwarded to spgl1.spgl1 (e.g. tolerances).

    Returns
    -------
    w_hat : torch.Tensor
        Estimated coefficient vector in the same shape as AT(y*0).
    """
    if spgl1_kwargs is None:
        spgl1_kwargs = {}

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

    # Allow user to override defaults if needed
    if "iter_lim" not in spgl1_kwargs:
        spgl1_kwargs["iter_lim"] = max_iter
    if verbose and "verbosity" not in spgl1_kwargs:
        spgl1_kwargs["verbosity"] = 2

    x_np, _, _, _ = spgl1(
        A_linop,
        y_np,
        tau=tau,
        sigma=sigma,
        x0=x0_np,
        **spgl1_kwargs,
    )

    w_hat = torch.from_numpy(x_np.astype(np.float32)).to(device).view_as(w0)
    return w_hat
