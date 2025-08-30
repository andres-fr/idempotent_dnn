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
import matplotlib.pyplot as plt


from idemp.utils import gaussian_noise, qr, lstsq


# ##############################################################################
# # HELPERS
# ##############################################################################
def gaussian_projector(
    dims, rank, orth=True, seed=None, dtype=torch.float64, device="cpu"
):
    """ """
    if rank > dims:
        raise ValueError("Rank can't be larger than number of rows/cols!")
    #
    left = gaussian_noise(
        (dims, rank),
        mean=0.0,
        std=1.0,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    #
    if orth:
        left = qr(left, in_place_q=True, return_r=False)
        result = left @ left.H, left
    if not orth:
        right = gaussian_noise(
            (rank, dims),
            mean=0.0,
            std=1.0,
            seed=seed + 1,
            dtype=dtype,
            device=device,
        )
        core = right @ left
        right = lstsq(core, right)
        result = left @ right, (left, right)
    #
    return result


def mutually_orthogonal_projectors(
    dims, ranks, seed=None, dtype=torch.float64, device="cpu"
):
    """ """
    if sum(ranks) != dims:
        raise ValueError("Sum of ranks must equal dims!")
    basis = gaussian_noise(
        (dims, dims),
        mean=0.0,
        std=1.0,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    basis = qr(basis, in_place_q=True, return_r=False)
    beg = 0
    projectors = []
    for r in ranks:
        end = beg + r
        projectors.append(basis[:, beg:end] @ basis[:, beg:end].H)
        beg = end
    #
    return projectors, basis


class CircularCorrelation:
    """Block-circulant and FFT correlation.

    This class implements 1D and 2D multi-channel circulant correlation (a.k.a.
    convolution in deep learning) via linear operators (either block-circulant
    or BCCB), as well as via FFT.

    The FFT implementation yields same results up to numerical precision.
    """

    @staticmethod
    def circulant1d(*matrices):
        """Builds a circulant matrix from a row of matrices.

        :param matrices: Collection of ``n`` matrices, each of shape
          ``(out_dims, in_dims)``.
        :returns: A matrix of shape ``(n*out_dims, n*in_dims)``, where each
          ``(i, j)`` block contains ``matrices[j - i]``.
        """
        shapes, dtypes, devices = zip(
            *((m.shape, m.dtype, m.device) for m in matrices)
        )
        if (len(set(shapes)), len(set(dtypes)), len(set(devices))) != (
            1,
            1,
            1,
        ):
            raise ValueError(
                "All matrices must be of same shape/dtype/device!"
            )
        h, w = shapes[0]
        c = len(matrices)
        #
        result = torch.empty(
            (c * h, c * w), dtype=dtypes[0], device=devices[0]
        )
        for i in range(c):
            for j in range(c):
                mat_idx = j - i
                beg_h, beg_w = i * h, j * w
                result[beg_h : beg_h + h, beg_w : beg_w + w] = matrices[
                    mat_idx
                ]
        #
        return result

    @classmethod
    def circorr1d(cls, x, kernel):
        """1D circular correlation.

        :param x: Tensor of shape ``(chans_in, n)``
        :param kernel: Tensor of shape ``(chans_out, chans_in, n)``
        :returns: Tensor of shape ``(chans_out, n)``, as the circular
          correlation between ``kernel`` and ``x``.

        Consider the case where ``n==1``: the kernel is a matrix that
        projects from ``chans_in`` to ``chans_out``. For larger ``n``, we have
        ``n`` linear projections, and the results are added.
        But since this is circulant, this project-and-add operation is also
        performed ``n`` times, in a rolling fashion. This is equivalent to
        building a block-circulant matrix of shape ``(chans_out*n, chans_in*n)``
        and performing a matrix-vector multiplication with flattened ``x``.
        Then, the output is reshaped to ``(chans_out, n)``, where each ``n``
        corresponds to one rolling position.
        """
        ch_out, ch_in, n = kernel.shape
        if x.shape != (ch_in, n):
            raise ValueError(
                f"Incompatible kernel and input shapes {kernel.shape, x.shape}"
            )
        #
        blocks = [kernel[:, :, i] for i in range(n)]
        circ = cls.circulant1d(*blocks)
        xflat = x.H.flatten()
        result = circ @ x.H.flatten()
        result = result.reshape(n, ch_out).H
        return result

    @classmethod
    def circorr1d_fft(cls, x, kernel):
        """FFT implementation of 1D circular correlation.

        In the FFT version, this operation diagonalizes to an elementwise
        multiplication between each ``fft(kernel[i, j])`` and ``fft(x[j])``,
        added over all j (which is asymptotically much cheaper). Note that
        instead of performing inverse FFT and adding over all input channels,
        we can first add the elementwise multiplications and then perform
        a single iFFT.
        """
        ch_out, ch_in, n = kernel.shape
        if x.shape != (ch_in, n):
            raise ValueError(
                f"Incompatible kernel and input shapes! {kernel.shape, x.shape}"
            )
        #
        kernel_f = torch.fft.fft(kernel)  # (ch_out, ch_in, n)
        x_f = torch.fft.fft(x, norm="ortho")  # (ch_in, n)
        result = (
            (kernel_f * x_f.unsqueeze(0).conj()).sum(1).conj()
        )  # (ch_out, n)
        result = torch.fft.ifft(result, norm="ortho")
        return result

    @classmethod
    def circulant2d(cls, kernel):
        """Builds a BCCB matrix from a row of circulant blocks.

        :param kernel: circorr2d parameters as tensor of shape
          ``(out, in, H, W)``.
        :returns: A matrix of shape ``(H*W*out, H*W*in)``, where each
          ``(hi, hj)`` block is a W-by-W circulant matrix corresponding to
          the multichannel circorr1d between the i-th kernel and the j-th
          input rows.

        Recall that multichannel 1D convolution amounts to a block-circulant
        linear operation. In the 2D setting, we flatten the input in a per-row
        basis, so we have ``H`` segments, each with ``W`` vectors of ``in``
        dimensions. The output will follow the exact same structure, but with
        ``H->W->out``blocks instead.

        Then, each one of the H-by-H main segments is a W-by-W block circulant
        matrix, corresponding to the circulant1d between the corresponding row
        of kernel and input.

        So, to construct this matrix, we first generate the ``H``
        block-circulant matrices, by picking the ``W`` segments of a given
        kernel row, and arranging them on a W-by-W circulant fashion.
        Doing this for each row yields the ``H`` elements that are used this
        time to build the block-circulant-with-circulant-blocks (BCCB) matrix.
        """
        ch_out, ch_in, H, W = kernel.shape
        #
        row_circs = []
        for h in range(H):
            row_blocks = kernel[:, :, h, :].permute(2, 0, 1)  # (W, out, in)
            row_circ = cls.circulant1d(*row_blocks)  # (W*out, W*in)
            row_circs.append(row_circ)
        #
        result = cls.circulant1d(*row_circs)
        return result

    @classmethod
    def circorr2d(cls, x, kernel):
        """2D circular correlation.

        :param x: Tensor of shape ``(chans_in, H, W)``
        :param kernel: Tensor of shape ``(chans_out, chans_in, H, W)``
        :returns: Tensor of shape ``(chans_out, H, W)``, as the circular
          correlation between ``kernel`` and ``x``.
        """
        ch_out, ch_in, H, W = kernel.shape
        if x.shape != (ch_in, H, W):
            raise ValueError(
                f"Incompatible kernel and input shapes {kernel.shape, x.shape}"
            )
        bccb = cls.circulant2d(kernel)  # (H*W*out, H*W*in)
        result = bccb @ x.reshape(ch_in, H * W).H.flatten()
        result = result.reshape(H, W, -1).permute(2, 0, 1)  # (out, H, W)
        return result

    @classmethod
    def circorr2d_fft(cls, x, kernel):
        """FFT implementation of 2D circular correlation.

        :param x: Tensor of shape (chans_in, H, W)
        :param kernel: Tensor of shape (chans_out, chans_in, H, W)
        :returns: Tensor of shape (chans_out, H, W)
        """
        ch_out, ch_in, H, W = kernel.shape
        if x.shape != (ch_in, H, W):
            raise ValueError(
                f"Incompatible kernel and input shapes {kernel.shape, x.shape}"
            )
        #
        kernel_f = torch.fft.fft2(kernel, dim=(-2, -1))  # (out, in, H, W)
        x_f = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")  # (in, H, W)
        result = (
            (kernel_f * x_f.unsqueeze(0).conj()).sum(1).conj()
        )  # (out, h, w)
        result = torch.fft.ifft2(
            result, dim=(-2, -1), norm="ortho"
        )  # (out, h, w)
        return result


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
    y1 = Conv1dPrototype.circorr1d(x, kernel)
    y2 = Conv1dPrototype.circorr1d_fft(x, kernel).real

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
    yy1 = Conv1dPrototype.circorr2d(xx, kernel2d)
    yy2 = Conv1dPrototype.circorr2d_fft(xx, kernel2d)
    breakpoint()
