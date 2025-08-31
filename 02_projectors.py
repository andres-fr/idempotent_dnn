#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:

* We have autograd-compatible, correct FFT convs
* In 02: train a small FFTconv 2D classifier on MNIST (make sure to pad)
  - So we show that our conv can be used for training
  - Also train a plain MLP
* In 03: MLP projector version:
  - Start with the proj version of the MLP. should be straightfwd
  - Then add the Cayley and kernel-bias constraints to update. How?
  - Idempotence should still hold, and should train well
* In 04: Conv projector version
  - Figure out how to apply this spectral-proj conv when input has variable length: params should still be in time domain?
  - Then add Cayley and kernel-bias; should train and be idemp
* In 05: Move onto larger nets and datasets for segmentation
  - If it trains and is idempotent, this is itself a success

If we reach this point, we will have trainable idempotent layers! what now?

1. Grab trained net and feed testset image: not OOD, but unseen.
  - wait, this way it can NEVER be inconsistent... or can it? maybe we should use losses? rethink


"""


import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from idemp.utils import gaussian_noise
from idemp.fftconv import circorr1d_fft, circorr2d_fft, CircularCorrelation
from idemp.projectors import gaussian_projector, mutually_orthogonal_projectors


# ##############################################################################
# # HELPERS
# ##############################################################################
class Circorr1d(torch.nn.Module):
    """ """

    def __init__(self, ch_in, ch_out, ksize, bias=True):
        """ """
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ksize = ksize
        self.kernel = torch.nn.Parameter(torch.randn(ch_out, ch_in, ksize))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(ch_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """ """
        b, c_in, n = x.shape
        if c_in != self.ch_in:
            raise ValueError(f"Expected {self.ch_in} channels, got {c_in}")
        if n < self.ksize:
            raise ValueError(f"Input must have at least {self.ksize} len!")
        #
        k = F.pad(self.kernel, (0, n - self.ksize))
        #
        k_f = torch.fft.fft(k, dim=-1)  # (out, in, n)
        x_f = torch.fft.fft(x, dim=-1, norm="ortho")  # (b, in, n)
        y = torch.einsum("oik,bik->bok", k_f, x_f.conj()).conj()  # (b, o, n)
        #
        y = torch.fft.ifft(y, dim=-1, norm="ortho").real  # (b, out, n)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)  # (b, out, n)
        #
        return y


class Circorr2d(torch.nn.Module):
    """ """

    def __init__(self, ch_in, ch_out, ksize, bias=True):
        """ """
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ksize = ksize
        self.kernel = torch.nn.Parameter(torch.randn(ch_out, ch_in, *ksize))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(ch_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """ """
        b, c_in, h, w = x.shape
        if c_in != self.ch_in:
            raise ValueError(f"Expected {self.ch_in} channels, got {c_in}")
        if h < self.ksize[0]:
            raise ValueError(f"Input must be larger than {self.ksize}!")
        if w < self.ksize[1]:
            raise ValueError(f"Input must be larger than {self.ksize}!")
        #
        k = F.pad(self.kernel, (0, w - self.ksize[1], 0, h - self.ksize[0]))
        #
        k_f = torch.fft.fft2(k, dim=(-2, -1))  # (out, in, H, W)
        x_f = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")  # (b, in, H, W)
        y = torch.einsum("ijhw,bjhw->bihw", k_f, x_f.conj()).conj()
        # (b, o, H, W)
        y = torch.fft.ifft2(y, dim=(-2, -1), norm="ortho").real  # (b, o, H, W)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)  # (b, out, H, W)
        #
        return y


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE, DEVICE = torch.float32, "cpu"
    # p1, _ = gaussian_projector(20, 5, orth=True, seed=12345)

    # (p1, p2, p3, p4), _ = mutually_orthogonal_projectors(
    #     20, (5, 5, 5, 5), seed=12345
    # )
    # v1 = gaussian_noise(20, seed=10000, dtype=torch.float64, device="cpu")
    # v2 = gaussian_noise(20, seed=10001, dtype=torch.float64, device="cpu")
    # v3 = gaussian_noise(20, seed=10002, dtype=torch.float64, device="cpu")
    # v4 = gaussian_noise(20, seed=10003, dtype=torch.float64, device="cpu")

    # ww = (p1 @ v1) + (p2 @ v2) + (p3 @ v3) + (p4 @ v4)
    # # torch.dist(p1 @ v1, p1 @ (p1 @ v1))
    # # (p2 @ (p1 @ v1)).norm()

    num_blocks, in_chans, out_chans = 3, 11, 5
    blocks = list(
        # gaussian_projector(
        #     chans,
        #     max(1, chans // 2),
        #     orth=True,
        #     seed=10000 + i,
        #     dtype=DTYPE,
        #     device=DEVICE,
        # )[0]
        gaussian_noise(
            (out_chans, in_chans), seed=10000 + i, dtype=DTYPE, device=DEVICE
        )
        for i in range(num_blocks)
    )
    kernel = (
        torch.hstack(blocks)
        .reshape(out_chans, num_blocks, in_chans)
        .permute(0, 2, 1)
    )
    x = gaussian_noise(
        (in_chans, num_blocks), seed=12345, dtype=DTYPE, device=DEVICE
    )
    y1 = CircularCorrelation.circorr1d(x, kernel)
    y2 = CircularCorrelation.circorr1d_fft(x, kernel).real

    #
    #
    #

    hh, ww = (13, 17)

    kernel2d = gaussian_noise(
        (out_chans, in_chans, hh, ww), seed=1234, dtype=DTYPE, device=DEVICE
    )
    xx = gaussian_noise(
        (in_chans, hh, ww), seed=1205, dtype=DTYPE, device=DEVICE
    )
    yy1 = CircularCorrelation.circorr2d(xx, kernel2d)
    yy2 = CircularCorrelation.circorr2d_fft(xx, kernel2d)

    """
    PLAN:

    At this point, we should be able to train e.g. a small logistic conv
    classifier on MNIST. This should be our 02 script

    At this point, is time to investigate projectors:
      - how is the equiv between proj-spectrum and circulant? can we convert?
      - is idempotence respected for arbitrary projs and kernel biases?

    As well as their gradients:
      - Does the projected grad respect the constraints?
      - Does it decrease the loss when trained?
      - Our 03 script should be a projector version of the 02 script, where each
        step tests an idempotence loss
      - We should also test a linear layer and conv1d.


    At this point, we have a repertoire of affine transforms that are idempotent
    and trainable via gradients. make a larger net and solve some task. Think
    about nonlinearities

    """

    # in this case we observe that our conv1d indeed performs dotprods as-is,
    # and shifts the kernel across the signal. x=(b, in, n), k=(out,in,n)
    x = torch.arange(10, dtype=DTYPE, device=DEVICE).reshape(2, 1, 5)
    k = torch.zeros((1, 1, 5), dtype=DTYPE, device=DEVICE)
    k[0, 0, 0] = 1
    # k[0, 0, 1] = 1
    y = circorr1d_fft(x, k).real

    # in the 2d case, same
    xx = (
        torch.arange(25, dtype=DTYPE, device=DEVICE).reshape(5, 5).unsqueeze(0)
    )
    xx = torch.stack([xx, xx * 2])
    kk = torch.zeros((1, 1, 5, 5), dtype=DTYPE, device=DEVICE)
    kk[0, 0, 2, 2] = 1
    yy = circorr2d_fft(xx, kk).real

    CC1 = Circorr1d(1, 1, 3, bias=True)
    CC1(x)

    CC2 = Circorr2d(1, 1, (3, 4), bias=True)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(
        CC2.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-2
    )
    for i in range(30000):
        opt.zero_grad()
        z = CC2(xx)
        loss = loss_fn(z, xx)
        loss.backward()
        opt.step()
        if i % 500 == 0:
            print(i, "loss:", loss)
    breakpoint()
