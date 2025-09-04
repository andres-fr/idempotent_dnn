#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:
* Finish linear + bias layers
* Define 4 nets for mnist recons: MLP/Conv with/without idempotent layers
* Show that all 4 train
* we got trainable idempotent convs
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

from idemp.projectors import (
    IdempotentLinear,
    IdempotentCircorr1d,
    IdempotentCircorr2d,
)


# ##############################################################################
# # HELPERS
# ##############################################################################


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE, DEVICE = torch.float32, "cpu"

    kk = IdempotentLinear(50, 5, bias=True)
    xx = gaussian_noise((7, 50), seed=1205, dtype=DTYPE, device=DEVICE)

    yy, dst = kk(xx)
    breakpoint()
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

    loss_fn = torch.nn.MSELoss()

    #
    #
    xx = (  # (1, 11, 5, 5)  bchw
        torch.arange(25, dtype=DTYPE, device=DEVICE)
        .view(1, 1, -1)
        .reshape(1, 1, 5, 5)
        .repeat(1, 11, 1, 1)
    )
    CC2 = IdempotentCircorr2d(11, (3, 3), 5, bias=True)
    opt = torch.optim.SGD(
        CC2.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-2
    )
    for i in range(40000):
        opt.zero_grad()
        z, dst = CC2(xx)
        loss = loss_fn(z, xx)  # + 0.01 * dst
        loss.backward()
        opt.step()
        if i % 500 == 0:
            print(i, "loss:", loss.item(), "dst:", dst.item())

    breakpoint()

    # plt.clf(); plt.plot(CC2.bias.detach().numpy()); plt.show()
    # plt.clf(); plt.plot(CC2.bias.detach().numpy()); plt.show()
    #
    #
    #
    CC2 = Circorr2d(1, 1, (3, 4), bias=True)
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
