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


def block_circulant(*matrices):
    """ """
    shapes, dtypes, devices = zip(
        *((m.shape, m.dtype, m.device) for m in matrices)
    )
    if (len(set(shapes)), len(set(dtypes)), len(set(devices))) != (1, 1, 1):
        raise ValueError("All matrices must be of same shape/dtype/device!")
    h, w = shapes[0]
    c = len(matrices)
    #
    result = torch.empty((c * h, c * w), dtype=dtypes[0], device=devices[0])
    for i in range(c):
        for j in range(c):
            mat_idx = (i - j) % c  # j - i
            beg_h, beg_w = i * h, j * w
            result[beg_h : beg_h + h, beg_w : beg_w + w] = matrices[mat_idx]
    #
    return result


def circorr(x, y, real=True):
    """ """
    x_fft = torch.fft.fft(x, norm="ortho")
    y_fft = torch.fft.fft(y)
    result = torch.fft.ifft((x_fft.conj() * y_fft), norm="ortho")
    result = result.real if real else result
    return result


def circorr_fft(x, kernel):
    """
    :param x: Tensor of shape ``(chans_in, n)``
    :param kernel: Tensor of shape ``(chans_out, chans_in, n)``
    :returns: Tensor of shape ``(chans_out, n)``, as the circular correlation
      between ``kernel`` and ``x``.

    Consider the case where ``n==1``. Here, the kernel is a matrix that
    projects from ``chans_in`` to ``chans_out``. For larger ``n``, we have
    ``n`` linear projections, and the results are added.
    But since this is circulant, this project-and-add operation is also
    performed ``n`` times, in a rolling fashion. This is equivalent to
    building a block-circulant matrix of shape ``(chans_out*n, chans_in*n)``
    and performing a matrix-vector multiplication with flattened ``x``.

    Now, in the FFT version, this turns out to diagonalize to an elementwise
    multiplication between each ``fft(kernel[i, j])`` and ``fft(x[j])``, added
    over all j (which is asymptotically much cheaper).
    """
    ch_out, ch_in, n = kernel.shape
    if x.shape != (ch_in, n):
        raise ValueError(
            f"Incompatible kernel and input shapes! {kernel.shape, x.shape}"
        )
    #
    kernel_f = torch.fft.fft(kernel, norm="ortho")  # (ch_out, ch_in, n)
    x_f = torch.fft.fft(x)  # (ch_in, n)
    result = (kernel_f * x_f.unsqueeze(0).conj()).sum(1).conj()  # (ch_out, n)
    result = torch.fft.ifft(result, norm="ortho")
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

    num_blocks, chans = 3, 10
    blocks = list(
        gaussian_projector(
            chans,
            max(1, chans // 2),
            orth=True,
            seed=10000 + i,
            dtype=DTYPE,
            device=DEVICE,
        )[0]
        # gaussian_noise(
        #     (chans, chans), seed=10000 + i, dtype=DTYPE, device=DEVICE
        # )
        for i in range(num_blocks)
    )
    circ = block_circulant(*blocks)
    vv = gaussian_noise(
        num_blocks * chans, seed=12345, dtype=DTYPE, device=DEVICE
    )
    ww = circ @ vv
    # torch.fft.ifft((torch.fft.fft(circ[0], norm="ortho") * torch.fft.fft(vv, norm="ortho")) , norm="ortho")

    # ww2 = torch.fft.ifft((torch.fft.fft(circ[0], norm="ortho").conj() * torch.fft.fft(vv)), norm="ortho" ).real

    # so this yields a num_blocks-dimensional vector, indexed by i/j
    # this is the thing that we circ-convolve with the j-th input chan, to
    # produce the i-th output chan.
    # circ[:10].reshape(10, 3, 10).permute(0, 2, 1)[0,  0]

    # so let's fix i/j, and compare the fft conv on the vec with the block:
    # as we can see, the sum of circorrs over j corresponds indeed to the i-th
    # output channel when doing the full matmul (ww):
    kernel = circ[:10].reshape(10, 3, 10).permute(0, 2, 1)
    vv_chans = vv.reshape(3, 10).H
    ww2 = circorr_fft(vv_chans, kernel).real.T.flatten()
    breakpoint()

    """
    NICE: we got the 1D FFT conv for multichannel.
    """
    # and the output chan i is the sum over all j:

    # at this point, conv_ij is correct

    # circ[:10].reshape(10, 3, 10)[0, :, 0]
    # NEW: CIRC FFT BLOCKCONV?
    breakpoint()

    # GPT APPROACH

    # A = circ[:chans].reshape(chans, chans, -1).permute(-1, 0, 1)
    A = torch.stack(blocks, dim=0)
    v = vv.reshape(chans, -1).permute(-1, 0)
    A_f = torch.fft.fft(A, n=num_blocks, dim=0)  # shape (n, m, m)
    v_f = torch.fft.fft(v, n=num_blocks, dim=0)  # shape (n, m)
    y_f = torch.einsum("nij,nj->ni", A_f, v_f)  # shape (n, m)
    y_blocks = torch.fft.ifft(y_f, n=num_blocks, dim=0)  # shape (n, m)
    # flatten back to vector
    y_fft = y_blocks.reshape(num_blocks * chans).real
    #
    #
    #
    T = torch.zeros((num_blocks * chans, num_blocks * chans), dtype=A.dtype)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block_idx = (i - j) % num_blocks
            T[
                i * chans : (i + 1) * chans, j * chans : (j + 1) * chans
            ] = blocks[block_idx]
    breakpoint()
    y_direct = T @ v

    breakpoint()

    ff1 = torch.fft.fft(
        circ[:chans].reshape(chans, chans, -1), norm="ortho"
    )  # (10, 10, 3)
    ff2 = torch.fft.fft(vv.reshape(chans, -1))
    breakpoint()

    circ[:chans].reshape(chans, chans, -1)

    ff1 = torch.fft.fft(circ[:chans].reshape(chans, chans, -1), norm="ortho")
    ff2 = torch.fft.fft(vv.reshape(chans, -1))
    mm = (ff1.permute(-1, 0, 1) @ ff2.permute(-1, 0).unsqueeze(-1)).squeeze(-1)

    breakpoint()

    # ppp, _ = gaussian_projector(chans, chans // 2, orth=True, seed=123, dtype=DTYPE, device=DEVICE)

    breakpoint()
