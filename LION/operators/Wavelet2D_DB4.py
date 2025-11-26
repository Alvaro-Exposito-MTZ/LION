"""2D orthogonal wavelet transform with periodised Daubechies wavelets.

TODO: This is not adjoint.
"""

from __future__ import annotations

import numpy as np
import pywt
import torch
from spyrit.core.torch import fwht as sfwht

from LION.operators import Operator


def fwht(x: torch.Tensor, dim):
    """2D fast Walsh-Hadamard transform over the last two dimensions."""
    x = sfwht(x, dim=dim[-1])  # width
    x = sfwht(x, dim=dim[-2])  # height
    return x


# class Wavelet2D:
#     """2D orthogonal wavelet transform with periodised Daubechies wavelets.

#     Parameters
#     ----------
#     shape : tuple of int
#         Image shape (H, W).
#     wavelet_name : str
#         Wavelet name, for example 'db4'.
#     level : int or None
#         Decomposition level. If None, uses a maximum valid level based on image size.
#     mode : str
#         Signal extension mode. 'periodization' gives an orthogonal transform.
#     device : str or torch.device
#         Device where transforms are allocated.
#     """

#     def __init__(
#         self,
#         shape,
#         wavelet_name: str = "db4",
#         level: int | None = None,
#         mode: str = "periodization",
#         device: str | torch.device = "cpu",
#     ):
#         self.shape = tuple(shape)
#         self.device = torch.device(device)
#         self.wavelet_name = wavelet_name
#         self.mode = mode

#         if level is None:
#             # Simple upper bound on number of dyadic levels
#             max_level = int(math.floor(math.log2(min(self.shape))))
#             self.level = max(1, max_level)
#         else:
#             self.level = int(level)

#         # Pytorch-wavelets modules
#         self.dwt = DWTForward(J=self.level, wave=self.wavelet_name, mode=self.mode).to(
#             self.device
#         )
#         self.idwt = DWTInverse(wave=self.wavelet_name, mode=self.mode).to(self.device)

#         # Build coefficient layout by running once on zeros
#         with torch.no_grad():
#             H, W = self.shape
#             x0 = torch.zeros(1, 1, H, W, device=self.device)
#             Yl, Yh_list = self.dwt(x0)

#         self.Yl_shape = tuple(Yl.shape)
#         self.Yh_shapes = [tuple(Yh.shape) for Yh in Yh_list]

#         self.size_Yl = int(Yl.numel())
#         self.size_Yh = [int(Yh.numel()) for Yh in Yh_list]
#         self.size = self.size_Yl + sum(self.size_Yh)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Wavelet analysis: image -> flat coefficient vector.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Real image, shape (H, W) or (1, 1, H, W).

#         Returns
#         -------
#         w : torch.Tensor
#             Flattened wavelet coefficient vector, shape (Nw,).
#         """
#         if x.dim() == 2:
#             x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
#         elif x.dim() == 4:
#             pass
#         else:
#             raise ValueError(f"Expected x with 2 or 4 dims, got shape {tuple(x.shape)}")

#         x = x.to(self.device)
#         Yl, Yh_list = self.dwt(x)

#         parts = [Yl.reshape(-1)]
#         for Yh in Yh_list:
#             parts.append(Yh.reshape(-1))

#         w = torch.cat(parts, dim=0)
#         return w

#     def inverse(self, w: torch.Tensor) -> torch.Tensor:
#         """Wavelet synthesis: flat coefficient vector -> image.

#         Parameters
#         ----------
#         w : torch.Tensor
#             Flattened wavelet coefficients, shape (Nw,).

#         Returns
#         -------
#         x : torch.Tensor
#             Reconstructed image in spatial domain, shape (H, W).
#         """
#         w = w.to(self.device)

#         start = 0
#         Yl_flat = w[start : start + self.size_Yl]
#         Yl = Yl_flat.view(self.Yl_shape)
#         start += self.size_Yl

#         Yh_list = []
#         for numel, shape in zip(self.size_Yh, self.Yh_shapes):
#             part_flat = w[start : start + numel]
#             Yh = part_flat.view(shape)
#             Yh_list.append(Yh)
#             start += numel

#         x = self.idwt((Yl, Yh_list))  # (1, 1, H, W) assumed

#         if x.dim() == 4 and x.size(0) == 1 and x.size(1) == 1:
#             x = x[0, 0]  # (H, W)

#         # Crop in case of small discrepancies
#         H, W = self.shape
#         return x[:H, :W]


class Wavelet2D_DB4(Operator):
    """2D orthogonal wavelet transform with periodised Daubechies wavelets.

    Parameters
    ----------
    shape : tuple of int
        Image shape (H, W).
    wavelet_name : str
        PyWavelets wavelet name, for example 'db4'.
    level : int or None
        Decomposition level. If None, uses the maximum valid level.
    mode : str
        Signal extension mode. 'periodization' gives an orthogonal transform.
    device : str or torch.device
        Device on which tensors are returned.
    """

    def __init__(
        self,
        shape,
        wavelet_name: str = "db4",
        level: int | None = None,
        mode: str = "periodization",
        device: str | torch.device = "cpu",
    ):
        self.shape = tuple(shape)
        self.device = torch.device(device)
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.mode = mode

        if level is None:
            max_level = pywt.dwt_max_level(min(self.shape), self.wavelet.dec_len)
            self.level = max_level
        else:
            self.level = int(level)

        # Build coefficient layout by running once on zeros
        x0 = np.zeros(self.shape, dtype=np.float32)
        coeffs = pywt.wavedec2(
            x0,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
        )
        arr, self.slices = pywt.coeffs_to_array(coeffs)
        self.arr_shape = arr.shape
        self.size = arr.size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wavelet analysis: image -> flat coefficient vector."""
        if x.dim() != 2:
            raise ValueError(
                f"Expected x with 2 dims (H, W), got shape {tuple(x.shape)}"
            )

        x_np = x.detach().cpu().numpy().astype(np.float32)

        # TODO: How to use the PyTorch backend and still ensure adjointness?
        coeffs = pywt.wavedec2(
            x_np,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
        )
        arr, _ = pywt.coeffs_to_array(coeffs)
        w = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        return w.reshape(-1)

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Wavelet synthesis: flat coefficient vector -> image."""
        w_np = w.detach().cpu().numpy().astype(np.float32)
        arr = w_np.reshape(self.arr_shape)
        coeffs = pywt.array_to_coeffs(arr, self.slices, output_format="wavedec2")

        # TODO: How to use the PyTorch backend and still ensure adjointness?
        x_np = pywt.waverec2(
            coeffs,
            wavelet=self.wavelet,
            mode=self.mode,
        )
        x_np = x_np.astype(np.float32)
        # Crop in case of off-by-one due to padding
        H, W = self.shape
        x_np = x_np[:H, :W]
        x = torch.from_numpy(x_np).to(self.device)
        return x

    def to(self, device: str | torch.device):
        """Move wavelet outputs to a given device."""
        self.device = torch.device(device)
        return self
