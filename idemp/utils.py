#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" """


import torch


# ##############################################################################
# # REPRODUCIBLE NOISE
# ##############################################################################
def integer_noise(
    shape, lo=0, hi=2, seed=None, dtype=torch.int64, device="cpu"
):
    """ """
    if seed is None:
        noise = torch.randint(
            low=lo, high=hi, size=shape, dtype=dtype, device=device
        )
    else:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        noise = torch.randint(
            low=lo,
            high=hi,
            size=shape,
            generator=rng,
            dtype=dtype,
            device=device,
        )
    #
    return noise
