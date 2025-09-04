#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
* we got trainable idempotent convs
  - add trainable idempotent biases
  - add error between FFT kernel and projected, optionally

* What about nonlinearities? Maybe the projectization is enough?


* Tenemos una conv layer que entrena
* Pero habia la duda del kernel size, que se resuelve con el straight-through

* Implementar el fwd/bwd para 1d y 2d, comprobar que es idempotente y entrenable para recons (al menos deberia converger a identity)
* Añadirle el bias, y mantener idempotence.
  - Intentar que entrene en algo dificil y ver q pasa

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
from idemp.fftconv import Circorr1d, Circorr2d
from idemp.projectors import gaussian_projector, mutually_orthogonal_projectors


# ##############################################################################
# # HELPERS
# ##############################################################################
class ProjStraightThrough(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(ctx, x, rank, saturation=1.0):
        """
        :param x: batch of arbitrary square matrices of shape ``(..., n, n)``.
        :param saturation: During backward step, eigenvalues are pushed to
          boolean via softmax. This parameter is the pre-softmax scaling. A
          larger saturation means closer to boolean: more similarity to a
          projector (gradients are les biased), but less quality of gradients.

        """
        h, w = x.shape[-2:]
        if h != w:
            raise ValueError(f"Only square matrices supported! {x.shape}")
        U, S, Vh = torch.linalg.svd(x)
        S_soft = torch.sigmoid(saturation * (S.T - S.T[rank]).T)
        ctx.save_for_backward(U, S_soft, Vh)
        # batch-wise outer product for projectors
        U_k = U[..., :rank]
        P = torch.matmul(U_k, U_k.transpose(-2, -1).conj())
        return P, U_k, S, S_soft

    @staticmethod
    def backward(ctx, grad_out):
        """ """
        U, S_soft, Vh = ctx.saved_tensors
        P_soft = (U * S_soft.unsqueeze(-2)) @ U.transpose(-2, -1).conj()
        grad_in = grad_out @ P_soft + P_soft @ grad_out
        return grad_in, None, None


class IdempotentCircorr1d(torch.nn.Module):
    """ """

    def __init__(self, chans, ksize, rank, bias=True, with_proj_dist=True):
        """ """
        super().__init__()
        self.chans = chans
        self.ksize = ksize
        self.rank = rank
        self.kernel = torch.nn.Parameter(torch.randn(chans, chans, ksize))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(chans))
            # self.bias.data.normal_(0.1)
        else:
            self.register_parameter("bias", None)
        self.with_proj_dist = with_proj_dist

    def forward(self, x):
        """ """
        b, c, n = x.shape
        if c != self.chans:
            raise ValueError(f"Expected {self.chans} channels, got {c}")
        if n < self.ksize:
            raise ValueError(f"Input must have at least {self.ksize} len!")
        #
        k = F.pad(self.kernel, (0, n - self.ksize))
        #
        x_f = torch.fft.fft(x, dim=-1, norm="ortho")  # (b, c, n)
        k_f = torch.fft.fft(k, dim=-1)  # (c, c, n)
        #
        k_f_proj = ProjStraightThrough.apply(  # (c, c, n)
            k_f.permute(2, 0, 1), self.rank, 1.0
        )[0].permute(1, 2, 0)
        #
        if k_f.numel() < (2 * x_f.numel()):
            y = torch.einsum("oin,bin->bon", k_f_proj.conj(), x_f)  # (b,c,n)
        else:
            y = torch.einsum("oin,bin->bon", k_f_proj, x_f.conj()).conj()
        #
        y = torch.fft.ifft(y, dim=-1, norm="ortho").real  # (b, c, n)
        if self.bias is not None:
            # this works because the FFT of the bias would be computed by
            # taking n copies and getting the FFT across the n axis. This
            # results in a real-valued DC component (scaled by n**0.5), and
            # the rest is zeros. Due to this identity up to scale, the desired
            # negproj across channels in the freq domain can be achieved in
            # the "plain" n domain
            bias_negproj = self.bias - k_f_proj[..., 0].real @ self.bias
            y = y + bias_negproj.view(-1, 1)  # (b, c, n)
        #
        dist = torch.dist(k_f, k_f_proj) if self.with_proj_dist else None
        return y, dist


class IdempotentCircorr2d(torch.nn.Module):
    """ """

    def __init__(self, chans, ksize, rank, bias=True, with_proj_dist=True):
        """ """
        super().__init__()
        self.chans = chans
        self.ksize = ksize
        self.rank = rank
        self.kernel = torch.nn.Parameter(torch.randn(chans, chans, *ksize))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(chans))
            self.bias.data.normal_(0.1)
        else:
            self.register_parameter("bias", None)
        self.with_proj_dist = with_proj_dist

    def forward(self, x):
        """ """
        b, c, h, w = x.shape
        if c != self.chans:
            raise ValueError(f"Expected {self.chans} channels, got {c}")
        if h < self.ksize[0]:
            raise ValueError(f"Input height must be >= {self.ksize}!")
        if w < self.ksize[1]:
            raise ValueError(f"Input width must be >= {self.ksize}!")
        #
        k = F.pad(self.kernel, (0, w - self.ksize[1], 0, h - self.ksize[0]))
        #
        x_f = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")  # (b, in, H, W)
        k_f = torch.fft.fft2(k, dim=(-2, -1))  # (out, in, H, W)

        k_f_proj = ProjStraightThrough.apply(
            k_f.permute(2, 3, 0, 1), self.rank, 1.0
        )[0].permute(2, 3, 0, 1)
        #
        if k_f.numel() < (2 * x_f.numel()):
            y = torch.einsum("oihw,bihw->bohw", k_f_proj.conj(), x_f)  # boHW
        else:
            y = torch.einsum("oihw,bihw->bohw", k_f_proj, x_f.conj()).conj()
        y = torch.fft.ifft2(y, dim=(-2, -1), norm="ortho").real  # (b, o, H, W)
        if self.bias is not None:
            # this works because the FFT of the bias would be computed by
            # taking n copies and getting the FFT across the n axis. This
            # results in a real-valued DC component (scaled by n**0.5), and
            # the rest is zeros. Due to this identity up to scale, the desired
            # negproj across channels in the freq domain can be achieved in
            # the "plain" n domain
            bias_negproj = self.bias - k_f_proj[..., 0, 0].real @ self.bias
            y = y + bias_negproj.view(-1, 1, 1)  # (b, c, H, W)
        #
        dist = torch.dist(k_f, k_f_proj) if self.with_proj_dist else None
        return y, dist


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

    # cc1d = Circorr1d(11, 5, 3, bias=False)
    # cc1d.kernel.data[:] = kernel
    # cc1d(x.unsqueeze(0))
    quack = IdempotentCircorr1d(11, 3, 5, bias=True)
    yy1, _ = quack(x.unsqueeze(0))
    yy2, _ = quack(yy1)
    # breakpoint()

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

    #
    #
    #
    quack = IdempotentCircorr2d(11, (3, 3), 5, bias=True)
    yyy1, dst1 = quack(xx.unsqueeze(0))
    yyy2, dst2 = quack(yyy1)

    breakpoint()

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
