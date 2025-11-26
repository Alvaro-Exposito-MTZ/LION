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
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from LION.classical_algorithms.compressed_sensing import (
    CompositeOp,
    debias_ls,
    fista_l1,
)
from LION.operators import PhotocurrentMapOp, Subsampler, Wavelet2D_DB4


def run_demo(
    dataset: torch.utils.data.Dataset,
    subtract_from_J: int = 1,
    delta_divided_by: int = 4,
    fista_lambda: float = 1e-3,  # needs tuning depending on noise level
    fista_max_iter: int = 300,
    debias_max_iter: int = 200,
    verbose: bool = False,
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
    H, W = 512, 512
    image_shape = (H, W)
    N = H * W
    J = 9  # 512x512 images
    N = 1 << J
    coarseJ = J - subtract_from_J
    delta = 1.0 / delta_divided_by

    # Wavelet transform Psi
    wavelet = Wavelet2D_DB4(image_shape, wavelet_name="db4", device=device)
    # wavelet = Wavelet2D_Haar(image_shape, wavelet_name="haar", device=device)

    # Photocurrent mapping operator Phi
    subsampler = Subsampler(n=N * N, coarseJ=coarseJ, delta=delta)
    phi = PhotocurrentMapOp(J=J, subsampler=subsampler, device=device)

    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, phi, device=device)

    # Measurements y (replace with real photocurrent data)
    y = phi.forward(im_tensor)

    # l1 reconstruction in wavelet domain
    print(
        "Running FISTA reconstruction: "
        f"{fista_max_iter} iterations, lambda={fista_lambda}..."
    )
    w_hat = fista_l1(
        A=A_op.forward,
        AT=A_op.adjoint,
        y=y,
        lam=fista_lambda,
        max_iter=fista_max_iter,
        tol=1e-4,
        L=None,
        verbose=verbose,
        progress_bar=True,
    )

    # Optional debiasing
    print(f"Running debiasing: {debias_max_iter} iterations...")
    w_debias = debias_ls(
        A=A_op.forward,
        AT=A_op.adjoint,
        y=y,
        w=w_hat,
        support_tol=1e-3,
        max_iter=debias_max_iter,
        tol=1e-5,
        progress_bar=True,
    )

    # Current map reconstruction
    cs_result_tensor = wavelet.inverse(w_debias)

    # Pseudo-inverse reconstruction (zero-filled)
    im_reconstructed_tensor = phi.pseudo_inv(y)  # warm-up

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
    # clim = None
    clim = [0, 1]
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
