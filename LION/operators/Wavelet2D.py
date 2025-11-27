"""2D orthogonal wavelet transform with periodised Daubechies wavelets.

TODO: This is not adjoint.
"""

from __future__ import annotations

import numpy as np
import pywt
import torch

from LION.operators.Operator import Operator


class Wavelet2D(Operator):
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
        super().__init__(device=device)
        self.shape = tuple(shape)
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

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward wavelet transform."""
        return self.forward(x, out=out)

    def forward(self, x: torch.Tensor, out=None) -> torch.Tensor:
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

    def adjoint(self, w: torch.Tensor, out=None) -> torch.Tensor:
        """Adjoint of the wavelet analysis operator."""
        # TODO: Test adjointness property
        return self.inverse(w)

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

    @property
    def domain_shape(self):
        return self.shape

    @property
    def range_shape(self):
        return self.arr_shape


class Wavelet2DNumpy(Operator):
    """2D orthogonal Daubechies 4 wavelet transform with periodised Daubechies wavelets for NumPy arrays.

    ::note::
        We prefer using the PyTorch version :class:`Wavelet2D_DB4` unless there is a specific reason
        to use NumPy.

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
    ):
        super().__init__(device=None)
        self.shape = tuple(shape)
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

    def __call__(self, x: np.ndarray, out=None) -> np.ndarray:
        """Apply the forward wavelet transform."""
        return self.forward(x, out=out)

    def forward(self, x: np.ndarray, out=None) -> np.ndarray:
        """Wavelet analysis: image -> flat coefficient vector."""
        if x.ndim != 2:
            raise ValueError(
                f"Expected x with 2 dims (H, W), got shape {tuple(x.shape)}"
            )

        # TODO: How to use the PyTorch backend and still ensure adjointness?
        coeffs = pywt.wavedec2(
            x,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
        )
        arr, _ = pywt.coeffs_to_array(coeffs)
        w = arr.astype(np.float32)
        return w.reshape(-1)

    def adjoint(self, w: np.ndarray, out=None) -> np.ndarray:
        """Adjoint of the wavelet analysis operator."""
        # TODO: Test adjointness property
        return self.inverse(w, out=out)

    def inverse(self, w: np.ndarray, out=None) -> np.ndarray:
        """Wavelet synthesis: flat coefficient vector -> image."""
        arr = w.reshape(self.arr_shape)
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
        return x_np

    @property
    def domain_shape(self):
        return self.shape

    @property
    def range_shape(self):
        return self.arr_shape
