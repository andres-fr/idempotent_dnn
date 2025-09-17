#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" """

import torch
import torch.nn.functional as F


# ##############################################################################
# # HELPERS
# ##############################################################################
def initialize_module(model, distr=torch.nn.init.xavier_uniform_, bias_val=0):
    """ """
    for pname, p in model.named_parameters():
        if "weight" in pname:
            distr(p)
        elif "bias" in pname:
            p.data[:] = bias_val
        else:
            # distr(p)
            raise NotImplementedError(f"Unknown weight: {pname}")


# ##############################################################################
# # LINEAR MLP
# ##############################################################################
class LinearMLP(torch.nn.Module):
    """Linear encodder with linear decoder."""

    def __init__(self, hidden_dims=350, logits=True):
        """ """
        super().__init__()
        self.logits = logits
        self.down = torch.nn.Linear(784, hidden_dims)
        self.up = torch.nn.Linear(hidden_dims, 784)
        initialize_module(self, torch.nn.init.xavier_uniform_, 0)

    def forward(self, x):
        """ """
        x = self.down(x)
        x = self.up(x)
        if not self.logits:
            x = F.sigmoid(x)
        return x


# ##############################################################################
# # MULTICHANNEL MLP
# ##############################################################################
class MultichannelMLP(torch.nn.Module):
    """Linear encoder with heavy multichannel decoder.

    This MLP autoencoder has a simple, linear encoder and a more complex,
    feedforward decoder that alternates pixel-layers with channel-layers:
    * In the first decoder layer, ``hdim`` copies of the ``zdim``-dimensional
      latent code are created, and for each copy, a linear layer is applied,
      producing ``chans`` features.
    * The following channel-layer contains, for each copy, a ``chan*chan``
      linear layer that acts as channel mixer
    * The following pixel-layer consists of a linear layer that is applied
      to each channel idependently, mapping from any number of copies
      (e.g. ``hdim``) to a different number of copies (e.g. 784).

    Alternating channel mixers with pixel mixers allows to overfit the FMNIST
    reconstruction task, while keeping a feedforward, linear MLP style and
    a very simple encoder.
    """

    def __init__(
        self,
        zdim=50,
        chans=50,
        xdim=784,
        hdim=350,
        with_1b=True,
        with_2b=True,
        activation=F.gelu,
        logits=False,
    ):
        """ """
        self.with_1b = with_1b
        self.with_2b = with_2b
        self.activation = activation
        self.logits = logits
        #
        super().__init__()
        self.enc = torch.nn.Linear(xdim, zdim)
        #
        self.dec_weight1 = torch.nn.Parameter(torch.randn(hdim, zdim, chans))
        self.dec_bias1 = torch.nn.Parameter(torch.randn(hdim, chans))
        #
        if with_1b:
            self.dec_weight1b = torch.nn.Parameter(
                torch.randn(hdim, chans, chans)
            )
            self.dec_bias1b = torch.nn.Parameter(torch.randn(hdim, chans))
        #
        self.dec_weight2 = torch.nn.Parameter(torch.randn(xdim, hdim, chans))
        self.dec_bias2 = torch.nn.Parameter(torch.randn(xdim, chans))
        #
        if with_2b:
            self.dec_weight2b = torch.nn.Parameter(
                torch.randn(xdim, chans, chans)
            )
            self.dec_bias2b = torch.nn.Parameter(torch.randn(xdim, chans))
        #
        self.dec_weight3 = torch.nn.Parameter(torch.randn(xdim, chans))
        self.dec_bias3 = torch.nn.Parameter(torch.randn(xdim))
        #
        initialize_module(self, torch.nn.init.xavier_uniform_, 0)

    def forward(self, x):
        """ """
        x = self.enc(x)  # b, zdim
        #
        x = torch.einsum("oic,bi->boc", self.dec_weight1, x) + self.dec_bias1
        x = self.activation(x)  # b, hdim, c
        #
        if self.with_1b:
            x = torch.einsum("nck,bnc->bnk", self.dec_weight1b, x)
            x = self.activation(x + self.dec_bias1b)  # b, hdim, c
        #
        x = torch.einsum("oic,bic->boc", self.dec_weight2, x) + self.dec_bias2
        x = self.activation(x)  # b, xdim, c
        #
        if self.with_2b:
            x = torch.einsum("nck,bnc->bnk", self.dec_weight2b, x)
            x = self.activation(x + self.dec_bias2b)  # b, xdim, c
        #
        x = torch.einsum("nc,bnc->bn", self.dec_weight3, x) + self.dec_bias3
        if not self.logits:
            x = F.sigmoid(x)  # b, xdim
        return x


class MultichannelPipeMLP(torch.nn.Module):
    """Linear encoder with heavy multichannel decoder.

    This MLP autoencoder has a simple, linear encoder and a more complex,
    feedforward decoder that alternates pixel-layers with channel-layers:
    * In the first decoder layer, ``hdim`` copies of the ``zdim``-dimensional
      latent code are created, and for each copy, a linear layer is applied,
      producing ``chans`` features.
    * The following channel-layer contains, for each copy, a ``chan*chan``
      linear layer that acts as channel mixer
    * The following pixel-layer consists of a linear layer that is applied
      to each channel idependently, mapping from any number of copies
      (e.g. ``hdim``) to a different number of copies (e.g. 784).

    Alternating channel mixers with pixel mixers allows to overfit the FMNIST
    reconstruction task, while keeping a feedforward, linear MLP style and
    a very simple encoder.
    """

    def __init__(
        self,
        zdim=50,
        xdim=784,
        hdim=100,
        activation=F.gelu,
        logits=False,
    ):
        """ """
        self.activation = activation
        self.logits = logits
        #
        super().__init__()
        self.enc = torch.nn.Linear(xdim, zdim)
        #
        self.dec_weight1 = torch.nn.Parameter(torch.randn(hdim, zdim, hdim))
        self.dec_bias1 = torch.nn.Parameter(torch.randn(hdim, hdim))
        #
        self.dec_weight2 = torch.nn.Parameter(torch.randn(hdim, hdim, hdim))
        self.dec_bias2 = torch.nn.Parameter(torch.randn(hdim, hdim))
        #
        self.dec_weight3 = torch.nn.Parameter(torch.randn(hdim, hdim, hdim))
        self.dec_bias3 = torch.nn.Parameter(torch.randn(hdim, hdim))
        #
        self.dec_weight4 = torch.nn.Parameter(torch.randn(hdim, hdim, hdim))
        self.dec_bias4 = torch.nn.Parameter(torch.randn(hdim, hdim))
        # # #
        # self.dec_weight5 = torch.nn.Parameter(torch.randn(hdim, hdim, hdim))
        # self.dec_bias5 = torch.nn.Parameter(torch.randn(hdim, hdim))
        #
        self.dec = torch.nn.Linear(hdim * hdim, xdim)
        #
        initialize_module(self, torch.nn.init.xavier_uniform_, 0)

    def forward(self, x):
        """ """
        x = self.enc(x)  # b, zdim
        # per-channel mapping
        x = torch.einsum("oic,bi->boc", self.dec_weight1, x) + self.dec_bias1
        x = self.activation(x)  # b, n, chans
        #
        x = torch.einsum("oic,bic->boc", self.dec_weight2, x) + self.dec_bias2
        x = self.activation(x)  # b, hdim, hdim
        #
        x = torch.einsum("oic,bic->boc", self.dec_weight3, x) + self.dec_bias3
        x = self.activation(x)  # b, hdim, hdim
        #
        x = torch.einsum("oic,bic->boc", self.dec_weight4, x) + self.dec_bias4
        x = self.activation(x)  # b, hdim, hdim
        # #
        # x = torch.einsum("oic,bic->boc", self.dec_weight5, x) + self.dec_bias5
        # x = self.activation(x)  # b, zdim, zdim
        #
        x = self.dec(x.reshape(len(x), -1))
        if not self.logits:
            x = F.sigmoid(x)  # b, xdim
        return x
