#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:

1. given a seed and a desired rank, create a random projector. Also bool for orth

2. For conv, if we have c_in -> c_out channels and a h*w kernel, we need to
   generate h*w projectors of shape (c_out, c_in), all of them orthogonal to each other.
3. Then, the operation P1@x1 + P2@x2 + ... is idempotent.

Doesn't work, because this operation admits N vectors and spits only 1.
So we would need some sort of symmetry, running it N times but then data extends to the margins...


For 1d multichan conv:

1. Run the regular torch layer and our custom impl as block-circulant with padding. Should be same
2. then, diagonalize block-circulant via ifft: should become block-diag
3.

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
            chans // 2,
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
