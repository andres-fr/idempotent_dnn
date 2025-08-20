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


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    # p1, _ = gaussian_projector(20, 5, orth=True, seed=12345)

    (p1, p2, p3, p4), _ = mutually_orthogonal_projectors(
        20, (5, 5, 5, 5), seed=12345
    )
    v1 = gaussian_noise(20, seed=10000, dtype=torch.float64, device="cpu")
    v2 = gaussian_noise(20, seed=10001, dtype=torch.float64, device="cpu")
    v3 = gaussian_noise(20, seed=10002, dtype=torch.float64, device="cpu")
    v4 = gaussian_noise(20, seed=10003, dtype=torch.float64, device="cpu")

    ww = (p1 @ v1) + (p2 @ v2) + (p3 @ v3) + (p4 @ v4)
    # torch.dist(p1 @ v1, p1 @ (p1 @ v1))
    # (p2 @ (p1 @ v1)).norm()
    breakpoint()
