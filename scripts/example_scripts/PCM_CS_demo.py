# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: lion_proposed
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo: Compressed Sensing Reconstruction for Photocurrent Mapping (PCM)


# %% [markdown]
# ## Imports
#

# %%
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from LION.classical_algorithms import fista_l1
from LION.classical_algorithms.spgl1 import spgl1_torch
from LION.operators import CompositeOp, PhotocurrentMapOp, Subsampler, Wavelet2D
from LION.operators.DebiasOp import debias_ls


def run_demo(
    dataset: torch.utils.data.Dataset,
    algo: Literal["fista", "spgl1"] = "fista",  # "fista" or "spgl1"
    subtract_from_J: int = 1,
    delta_divided_by: int = 4,
    lam: float = 1e-3,
    max_iter: int = 300,
    tol: float = 1e-4,
    debias_max_iter: int = 200,
    debias_support_tol: float = 1e-3,
    debias_tol: float = 1e-5,
    verbose: bool = False,
    clim: tuple | None = None,
):
    device = torch.get_default_device()
    print(f"Using device: {device}")
    # %%
    sino, target = dataset[0]
    im_tensor = target.unsqueeze(0)  # (1,1,H,W)
    # Normalise image to [0,1]
    im_tensor = (im_tensor - im_tensor.min()) / (im_tensor.max() - im_tensor.min())
    im_tensor = im_tensor.to(device)

    # Image size
    J = 9  # 512x512 images
    H = W = 1 << J
    coarseJ = J - subtract_from_J
    delta = 1.0 / delta_divided_by

    # Wavelet transform Psi
    wavelet = Wavelet2D((H, W), wavelet_name="db4", device=device)

    # Photocurrent mapping operator Phi
    subsampler = Subsampler(n=H * W, coarseJ=coarseJ, delta=delta)
    phi = PhotocurrentMapOp(J=J, subsampler=subsampler, device=device)

    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, phi, device=device)

    # Measurements y (replace with real photocurrent data)
    y = phi(im_tensor)

    # l1 reconstruction in wavelet domain
    if algo == "fista":
        print(
            "Running FISTA reconstruction: " f"{max_iter} iterations, lambda={lam}..."
        )
        w_hat = fista_l1(
            op=A_op,
            y=y,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
            L=None,
            verbose=verbose,
            progress_bar=True,
        )
    elif algo == "spgl1":
        print(
            "Running SPGL1 reconstruction: " f"{max_iter} iterations, lambda={lam}..."
        )
        verbosity = 2 if verbose else 0
        # (LASSO interpretation: lam -> tau)
        w_hat = spgl1_torch(
            op=A_op,
            y=y,
            lam=lam,  # l1 budget
            iter_lim=max_iter,
            mode="lasso",  # or "bpdn" if lam should be a noise bound
            verbosity=verbosity,
            opt_tol=tol,
        )
    else:
        raise ValueError(f"Unknown algo '{algo}', expected 'fista' or 'spgl1'.")

    # Optional debiasing
    print(f"Running debiasing: {debias_max_iter} iterations...")
    w_debias = debias_ls(
        op=A_op,
        y=y,
        w=w_hat,
        support_tol=debias_support_tol,
        max_iter=debias_max_iter,
        tol=debias_tol,
        progress_bar=True,
    )

    # Current map reconstruction
    cs_result_tensor = wavelet.inverse(w_debias)

    # Pseudo-inverse reconstruction (zero-filled)
    im_reconstructed_tensor = phi.pseudo_inv(y)

    # %%
    im_np = im_tensor.squeeze().cpu().numpy()
    cs_result_np = cs_result_tensor.squeeze().detach().cpu().numpy()
    im_reconstructed_np = im_reconstructed_tensor.squeeze().detach().cpu().numpy()

    data_range = im_np.max() - im_np.min()

    psnr_zf = psnr(im_np, im_reconstructed_np, data_range=data_range)
    psnr_cs = psnr(im_np, cs_result_np, data_range=data_range)

    ssim_zf = ssim(im_np, im_reconstructed_np, data_range=data_range)
    ssim_cs = ssim(im_np, cs_result_np, data_range=data_range)

    # %%
    n_subplots = 4
    plt.figure(figsize=(n_subplots * 4, 4))

    plt.subplot(1, n_subplots, 1)
    plt.imshow(im_np, cmap="gray", clim=clim)
    plt.title("Original Image")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1, n_subplots, 2)
    plt.imshow(im_reconstructed_np, cmap="gray", clim=clim)
    plt.title(
        f"Zero-filled Reconstruction\nPSNR: {psnr_zf:.2f} dB, SSIM: {ssim_zf:.4f}"
    )
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1, n_subplots, 3)
    plt.imshow(cs_result_np, cmap="gray", clim=clim)
    plt.title(f"CS Reconstruction\nPSNR: {psnr_cs:.2f} dB, SSIM: {ssim_cs:.4f}")
    plt.axis("off")
    plt.colorbar()

    plt.suptitle(
        "PCM Reconstructions Comparison\n"
        + f"sampling rate: {delta * 100:.2f}%, in-order measurements: {1 / (1 << (subtract_from_J * 2)) * 100:.2f}%",
        x=0.4,
        y=-0.05,
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig("pcm_reconstructions_comparison.png", dpi=150)
    plt.show()
