#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" """

import torch
import torch.nn.functional as F


# ##############################################################################
# # FFT BATCHED CIRCULAR CORRELATION (FUNCTIONS AND TORCH MODULES)
# ##############################################################################
def circorr1d_fft(x, kernel):
    """FFT batched 1D circular correlation.

    :param x: Tensor of shape ``(batch, chans_in, n)``
    :param kernel: Tensor of shape ``(chans_out, chans_in, n)``
    :returns: Tensor of shape ``(batch, chans_out, n)``, as the circular
      correlation between ``kernel`` and ``x``.
    """
    ch_out, ch_in, n = kernel.shape
    if x.shape[1:] != (ch_in, n):
        raise ValueError(
            f"Incompatible kernel and input shapes! {kernel.shape, x.shape}"
        )
    #
    kernel_f = torch.fft.fft(kernel)  # (ch_out, in, n)
    x_f = torch.fft.fft(x, norm="ortho")  # (b, in, n)
    #
    y = torch.einsum("ijk,bjk->bik", kernel_f, x_f.conj()).conj()  # (b, out,n)
    y = torch.fft.ifft(y, norm="ortho")  # (b, out, n)
    return y


def circorr2d_fft(x, kernel):
    """FFT batched 2D circular correlation.

    :param x: Tensor of shape (batch, chans_in, H, W)
    :param kernel: Tensor of shape (chans_out, chans_in, H, W)
    :returns: Tensor of shape (batch, chans_out, H, W)
    """
    ch_out, ch_in, H, W = kernel.shape
    if x.shape[1:] != (ch_in, H, W):
        raise ValueError(
            f"Incompatible kernel and input shapes {kernel.shape, x.shape}"
        )
    #
    k_f = torch.fft.fft2(kernel, dim=(-2, -1))  # (out, in, H, W)
    x_f = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")  # (b, in, H, W)
    y = torch.einsum("ijhw,bjhw->bihw", k_f, x_f.conj()).conj()  # (b, o, H, W)
    y = torch.fft.ifft2(y, dim=(-2, -1), norm="ortho")  # (b, out, H, W)
    return y


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
        if k_f.numel() < (2 * x_f.numel()):
            y = torch.einsum("oin,bin->bon", k_f.conj(), x_f)  # (b, out, n)
        else:
            y = torch.einsum("oin,bin->bon", k_f, x_f.conj()).conj()
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
        if k_f.numel() < (2 * x_f.numel()):
            y = torch.einsum("oihw,bihw->bohw", k_f.conj(), x_f)  # (b,o,H,W)
        else:
            y = torch.einsum("oihw,bihw->bohw", k_f, x_f.conj()).conj()
        y = torch.fft.ifft2(y, dim=(-2, -1), norm="ortho").real  # (b, o, H, W)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)  # (b, out, H, W)
        #
        return y


# ##############################################################################
# # TESTING PROTOTYPE
# ##############################################################################
class CircularCorrelation:
    """Sample implementation of block-circulant and FFT correlation.

    This class implements 1D and 2D multi-channel circulant correlation (a.k.a.
    convolution in deep learning) via linear operators (either block-circulant
    or BCCB), as well as via FFT.

    It is mainly intended to show that the FFT implementation yields
    same results up to numerical precision.

    The following usage examples illustrate the circulant behaviour::

      # 1-dimensional:
      # x=[0,1,2,3,4,5,6,7,8,9,10]
      # y=[1,2,3,4,5,6,7,8,9,10,0]
      x = torch.arange(11, dtype=dt, device=dv).unsqueeze(0)
      k = torch.zeros((1, 1, 11), dtype=dt, device=dv)
      k[0, 0, 1] = 1
      y = CircularCorrelation.circorr1d_fft(x, k).real

      # 2-dimensional:
      # xx = [[0...4], [5...9], [10...14], [15...19], [20...24]]
      # yy = [[12,13,14,10,11], [17,18,19,15,16], [22,23,24,20,21],[2,3,4,0,1],
              [7,8,9,5,6]]
      xx = (
          torch.arange(25, dtype=dt, device=dv).reshape(5, 5).unsqueeze(0)
      )
      kk = torch.zeros((1, 1, 5, 5), dtype=dt, device=dv)
      kk[0, 0, 2, 2] = 1
      yy = CircularCorrelation.circorr2d_fft(xx, kk).real
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
