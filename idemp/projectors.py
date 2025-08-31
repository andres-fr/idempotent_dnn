#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" """


from .utils import gaussian_noise, qr, lstsq
import torch


# ##############################################################################
# # LINEAR
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
# # CONVOLUTIONAL
# ##############################################################################
