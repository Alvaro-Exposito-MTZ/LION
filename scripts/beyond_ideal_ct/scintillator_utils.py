# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Alvaro Exposito Martinez
# =============================================================================
# 1D PRF estimation and deconvolution of 2DeteCT sinograms.
# Based on 2detect.ipynb and prf.py from Graas & Lucka (2025),
# "Scintillator decorrelation for self-supervised X-ray radiograph denoising",
# https://github.com/adriaangraas/scintillatordecorrelator

import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import sqrtm
from scipy.signal import correlate


def estimate_autocorrelation(data, out_len=5):
    """1D autocorrelation of the noise along the detector.
    data: (slices, angles, pixels), a region without object."""
    # stabilize the noise, i.e., divide by its variance
    # for poisson noise, the variance equals the mean
    WD = np.copy(data)
    for p in range(data.shape[2]):
        for s in range(data.shape[0]):
            V = np.mean(data[s, :, p])
            WD[s, :, p] = (data[s, :, p] - V) / np.sqrt(V)

    # now, all slices and sinogram rows have the same variance
    # so we concat those
    WD = np.concatenate(WD, axis=0)

    # 1D autocorrelation, only with real pairs of pixels (no np.roll)
    n = WD.shape[1]
    out = np.zeros(out_len, dtype=np.float64)
    for i in range(out_len):
        shift = abs(i - out_len // 2)
        out[i] = np.mean(WD[:, : n - shift] * WD[:, shift:])

    return out


def convolution_matrix(kernel, image_shape):
    """From prf.py."""
    height, width = image_shape
    operator = np.zeros((height * width, height * width))
    for i in range(height):
        for j in range(width):
            basis_image = np.zeros(image_shape)
            basis_image[i, j] = 1
            convolved_image = correlate(basis_image, kernel, mode="same")
            operator[:, i * width + j] = convolved_image.flatten()
    return operator


def solve_kernel(h_conv_h):
    """From prf.py."""
    out_shape = list((s // 2 + 1 for s in h_conv_h.shape))
    op = convolution_matrix(h_conv_h, out_shape)
    H = sqrtm(op)
    h = np.real(H[op.shape[0] // 2]).reshape(out_shape)
    return h


class Deconvolve1D:
    """Deconvolve the data in Fourier space using a point-response function."""

    def __init__(self, prf, padding=20):
        assert prf.ndim == 1
        self.prf = prf
        self.padding = padding
        self._otf = None
        self._otf_shape = None
        self._fact = None

    def __call__(self, im):
        assert im.ndim == 1
        # pad the row, so the FFT does not mix both ends of the detector
        n = im.shape[0]
        im = np.pad(im, self.padding, mode="edge")

        if self._otf_shape is None or im.shape != self._otf_shape:
            self._otf = fft(self.prf, im.shape[0])
            self._otf_shape = im.shape
            self._fact = np.conj(self._otf) / (np.abs(self._otf) ** 2)

        out = np.real(ifft(fft(im) * self._fact))
        out = np.roll(out, shift=self.prf.shape[0] // 2, axis=0)
        return out[self.padding : self.padding + n]
