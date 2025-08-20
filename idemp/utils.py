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


def gaussian_noise(
    shape, mean=0.0, std=1.0, seed=None, dtype=torch.float64, device="cpu"
):
    """Reproducible ``torch.normal`` Gaussian noise.

    :returns: A tensor of given shape, dtype and device, containing gaussian
      noise with given mean and std (analogous to ``torch.normal``), but with
      reproducible behaviour fixed to given random seed.
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    #
    noise = torch.zeros(shape, dtype=dtype, device=device)
    noise.normal_(mean=mean, std=std, generator=rng)
    return noise


# ##############################################################################
# # MATRIX ROUTINE WRAPPERS
# ##############################################################################
def qr(A, in_place_q=False, return_r=False):
    """Thin QR-decomposition of given matrix.

    :param A: Matrix to orthogonalize, needs to be compatible with either
      ``scipy.linalg.qr`` or ``torch.linalg.qr``. It must be square or tall.
    :param in_place_q: If true, ``A[:] = Q`` will be performed.
    :returns: If ``return_R`` is true, returns ``(Q, R)`` such that ``Q``
      has orthonormal columns, ``R`` is upper triangular and ``A = Q @ R``
      as per the QR decomposition. Otherwise, returns just ``Q``.
    """
    h, w = A.shape
    if h < w:
        raise ValueError("Only non-fat matrices supported!")
    #
    if isinstance(A, torch.Tensor):
        Q, R = torch.linalg.qr(A, mode="reduced")
    else:
        # TODO: support pivoting in all modalities
        Q, R = scipy.linalg.qr(A, mode="economic", pivoting=False)
    #
    if in_place_q:
        A[:] = Q
        Q = A
    if return_r:
        return Q, R
    else:
        return Q


def lstsq(A, b, rcond=1e-6):
    """Least-squares solver.

    :returns: ``x`` such that ``frob(Ax - b)`` is minimized.
    """
    if isinstance(A, torch.Tensor):
        # do not use default gelsy driver: nondeterm results yielding errors
        driver = "gels" if b.device.type == "cuda" else "gelsd"
        result = torch.linalg.lstsq(A, b, rcond=rcond, driver=driver).solution
    else:
        result = scipy.linalg.lstsq(A, b, cond=rcond, lapack_driver="gelsd")[0]
    return result
