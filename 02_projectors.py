#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:

1. Now we have multichan conv, compare to torch.
  - Check: torch conv should work same
  - Goal: create a FftConv layer where we can specify the OrthProj freq response
  - Bonus: translate to regular conv. How?

2. Check that applying this conv several times results in same as once

3. Use it to train sth, extend to 2d.

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


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE, DEVICE = torch.float64, "cpu"
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
    investigate with deltas. plan is to get same behaviour as torch 1d and 2d
    up to padding etc. Then, leave the prototype as-is and write a conv FFT
    function with an interface similar to torch for drop-in replacement (batched and kbias)

    Then, also profile the FFT version, which should have minimal memory and
    runtime while being autodiffable.

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
    breakpoint()

    # now pytorch 1d:
    # F.conv1d(x.unsqueeze(0), k, padding=0)

    breakpoint()
