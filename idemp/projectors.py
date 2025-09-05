#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" """


from .utils import gaussian_noise, qr, lstsq
import torch
import torch.nn.functional as F


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


class ProjStraightThrough(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(ctx, x, rank, saturation=1.0):
        """
        :param x: batch of arbitrary square matrices of shape ``(..., n, n)``.
        :param saturation: During backward step, eigenvalues are pushed to
          boolean via softmax. This parameter is the pre-softmax scaling. A
          larger saturation means closer to boolean: more similarity to a
          projector (gradients are les biased), but less quality of gradients.

        """
        h, w = x.shape[-2:]
        if h != w:
            raise ValueError(f"Only square matrices supported! {x.shape}")
        U, S, Vh = torch.linalg.svd(x)
        if len(S.shape) == 1:
            S_soft = torch.sigmoid(saturation * (S - S[rank]))
        elif len(S.shape) == 2:
            S_soft = torch.sigmoid(saturation * (S.T - S.T[rank]).T)
        else:
            S_soft = torch.sigmoid(saturation * (S.T - S.T[rank]).T)
            # raise NotImplementedError("Get rid of transpose!")
        ctx.save_for_backward(U, S_soft, Vh)
        # batch-wise outer product for projectors
        U_k = U[..., :rank]
        P = torch.matmul(U_k, U_k.transpose(-2, -1).conj())
        return P  # , U_k, S, S_soft

    @staticmethod
    def backward(ctx, grad_out):
        """ """
        # grad_out seems to be of same shape as P (..., n, n)
        U, S_soft, Vh = ctx.saved_tensors
        P_soft = (U * S_soft.unsqueeze(-2)) @ U.transpose(-2, -1).conj()
        grad_in = grad_out @ P_soft + P_soft @ grad_out  # (n,n)@(n,n) matmul
        return grad_in, None, None


# ##############################################################################
# # LINEAR IDEMPOTENT LAYER
# ##############################################################################
class IdempotentLinear(torch.nn.Module):
    """ """

    def __init__(
        self, dims, rank, bias=True, with_proj_dist=True, saturation=1.0
    ):
        """ """
        super().__init__()
        self.dims = dims
        self.rank = rank
        self.saturation = saturation
        self.weight = torch.nn.Parameter(torch.randn(dims, dims))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(dims))
            # self.bias.data.normal_(0.1)
        else:
            self.register_parameter("bias", None)
        self.with_proj_dist = with_proj_dist

    def forward(self, x):
        """ """
        b, n = x.shape
        if n != self.dims:
            raise ValueError(f"Expected {self.dims} dims, got {n}")
        w_proj = ProjStraightThrough.apply(
            self.weight, self.rank, self.saturation
        )
        x = torch.einsum("oi,bi->bo", w_proj, x)
        if self.bias is not None:
            bias_negproj = self.bias - w_proj @ self.bias
            x = x + bias_negproj
        #
        dist = torch.dist(self.weight, w_proj) if self.with_proj_dist else None
        return x, dist


# ##############################################################################
# # CONV IDEMPOTENT LAYERS
# ##############################################################################
class IdempotentCircorr1d(torch.nn.Module):
    """ """

    def __init__(
        self,
        chans,
        ksize,
        rank,
        bias=True,
        with_proj_dist=True,
        saturation=1.0,
    ):
        """ """
        super().__init__()
        self.chans = chans
        self.ksize = ksize
        self.rank = rank
        self.saturation = saturation
        self.kernel = torch.nn.Parameter(torch.randn(chans, chans, ksize))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(chans))
            # self.bias.data.normal_(0.1)
        else:
            self.register_parameter("bias", None)
        self.with_proj_dist = with_proj_dist

    def forward(self, x):
        """ """
        b, c, n = x.shape
        if c != self.chans:
            raise ValueError(f"Expected {self.chans} channels, got {c}")
        if n < self.ksize:
            raise ValueError(f"Input must have at least {self.ksize} len!")
        #
        k = F.pad(self.kernel, (0, n - self.ksize))
        #
        x_f = torch.fft.fft(x, dim=-1, norm="ortho")  # (b, c, n)
        k_f = torch.fft.fft(k, dim=-1)  # (c, c, n)
        #
        k_f_proj = ProjStraightThrough.apply(  # (c, c, n)
            k_f.permute(2, 0, 1), self.rank, self.saturation
        ).permute(1, 2, 0)
        #
        if k_f.numel() < (2 * x_f.numel()):
            y = torch.einsum("oin,bin->bon", k_f_proj.conj(), x_f)  # (b,c,n)
        else:
            y = torch.einsum("oin,bin->bon", k_f_proj, x_f.conj()).conj()
        #
        y = torch.fft.ifft(y, dim=-1, norm="ortho").real  # (b, c, n)
        if self.bias is not None:
            # this works because the FFT of the bias would be computed by
            # taking n copies and getting the FFT across the n axis. This
            # results in a real-valued DC component (scaled by n**0.5), and
            # the rest is zeros. Due to this identity up to scale, the desired
            # negproj across channels in the freq domain can be achieved in
            # the "plain" n domain
            bias_negproj = self.bias - k_f_proj[..., 0].real @ self.bias
            y = y + bias_negproj.view(-1, 1)  # (b, c, n)
        #
        dist = torch.dist(k_f, k_f_proj) if self.with_proj_dist else None
        return y, dist


class IdempotentCircorr2d(torch.nn.Module):
    """ """

    def __init__(
        self,
        chans,
        ksize,
        rank,
        bias=True,
        with_proj_dist=True,
        saturation=1.0,
    ):
        """ """
        super().__init__()
        self.chans = chans
        self.ksize = ksize
        self.rank = rank
        self.saturation = saturation
        self.kernel = torch.nn.Parameter(torch.randn(chans, chans, *ksize))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(chans))
            # self.bias.data.normal_(0.1)
        else:
            self.register_parameter("bias", None)
        self.with_proj_dist = with_proj_dist

    def forward(self, x):
        """ """
        b, c, h, w = x.shape
        if c != self.chans:
            raise ValueError(f"Expected {self.chans} channels, got {c}")
        if h < self.ksize[0]:
            raise ValueError(f"Input height must be >= {self.ksize}!")
        if w < self.ksize[1]:
            raise ValueError(f"Input width must be >= {self.ksize}!")
        #
        k = F.pad(self.kernel, (0, w - self.ksize[1], 0, h - self.ksize[0]))
        #
        x_f = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")  # (b, in, H, W)
        k_f = torch.fft.fft2(k, dim=(-2, -1))  # (out, in, H, W)

        k_f_proj = ProjStraightThrough.apply(
            k_f.permute(2, 3, 0, 1), self.rank, self.saturation
        ).permute(2, 3, 0, 1)
        #
        if k_f.numel() < (2 * x_f.numel()):
            y = torch.einsum("oihw,bihw->bohw", k_f_proj.conj(), x_f)  # boHW
        else:
            y = torch.einsum("oihw,bihw->bohw", k_f_proj, x_f.conj()).conj()
        y = torch.fft.ifft2(y, dim=(-2, -1), norm="ortho").real  # (b, o, H, W)
        if self.bias is not None:
            # this works because the FFT of the bias would be computed by
            # taking n copies and getting the FFT across the n axis. This
            # results in a real-valued DC component (scaled by n**0.5), and
            # the rest is zeros. Due to this identity up to scale, the desired
            # negproj across channels in the freq domain can be achieved in
            # the "plain" n domain
            bias_negproj = self.bias - k_f_proj[..., 0, 0].real @ self.bias
            y = y + bias_negproj.view(-1, 1, 1)  # (b, c, H, W)
        #
        dist = torch.dist(k_f, k_f_proj) if self.with_proj_dist else None
        return y, dist
