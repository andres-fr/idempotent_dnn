#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:

* first and foremost, get a freaking MLP that beats PCA
* then, run the analogous idempotent as up @ [idemp/activ]  @ down
  - should also beat PCA and also be good

* What is the role of idemp here?
  - in PCA we got up @ down, as top-k eigenvectors
  - here, we got  up @ [idemp/activ] @ down
  - so encoding is
*


* Finish linear + bias layers
* Define 4 nets for mnist recons: MLP/Conv with/without idempotent layers
* Show that all 4 train
* we got trainable idempotent convs
* What about nonlinearities? Maybe the projectization is enough?


* Tenemos una conv layer que entrena
* Pero habia la duda del kernel size, que se resuelve con el straight-through

* Implementar el fwd/bwd para 1d y 2d, comprobar que es idempotente y entrenable para recons (al menos deberia converger a identity)
* Añadirle el bias, y mantener idempotence.
  - Intentar que entrene en algo dificil y ver q pasa

TODO:

* We have autograd-compatible, correct FFT convs
* In 02: train a small FFTconv 2D classifier on MNIST (make sure to pad)
  - So we show that our conv can be used for training
  - Also train a plain MLP
* In 03: MLP projector version:
  - Start with the proj version of the MLP. should be straightfwd
  - Then add the Cayley and kernel-bias constraints to update. How?
  - Idempotence should still hold, and should train well
* In 04: Conv projector version
  - Figure out how to apply this spectral-proj conv when input has variable length: params should still be in time domain?
  - Then add Cayley and kernel-bias; should train and be idemp
* In 05: Move onto larger nets and datasets for segmentation
  - If it trains and is idempotent, this is itself a success

If we reach this point, we will have trainable idempotent layers! what now?

1. Grab trained net and feed testset image: not OOD, but unseen.
  - wait, this way it can NEVER be inconsistent... or can it? maybe we should use losses? rethink


"""


import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from idemp.utils import gaussian_noise
from idemp.fftconv import circorr1d_fft, circorr2d_fft, CircularCorrelation
from idemp.fftconv import Circorr1d, Circorr2d
from idemp.projectors import gaussian_projector, mutually_orthogonal_projectors

from idemp.projectors import (
    IdempotentLinear,
    IdempotentCircorr1d,
    IdempotentCircorr2d,
)


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


# class IdempotentMLP(torch.nn.Module):
#     """
#     This one works well on FMNIST trained for 15 epochs with BCELoss and
#     Adam(lr=1e-3, wd=1e-5). 0.25 loss is already decent
#     """

#     def __init__(self, hidden_dims=350, rank=150, activation=torch.nn.Tanh):
#         """ """
#         super().__init__()
#         self.down = torch.nn.Linear(784, hidden_dims)
#         self.h1 = IdempotentLinear(hidden_dims, rank, bias=True, saturation=2)
#         self.a1 = activation()
#         self.up = torch.nn.Linear(hidden_dims, 784)

#     def forward(self, x):
#         """ """
#         dsts = []
#         x = self.down(x)
#         x, dst1 = self.h1(x)
#         dsts.append(dst1)
#         x = self.a1(x)
#         x = self.up(x)
#         return x, dsts


class PCA_MLP(torch.nn.Module):
    """ """

    def __init__(self, hidden_dims=350):
        """ """
        super().__init__()
        self.down = torch.nn.Linear(784, hidden_dims)
        self.up = torch.nn.Linear(hidden_dims, 784)
        initialize_module(self, torch.nn.init.xavier_uniform_, 0)

    def forward(self, x):
        """ """
        dsts = []
        x = self.down(x)
        x = self.up(x)
        return x, dsts


class MultichanMLP(torch.nn.Module):
    """

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
        zdim=350,
        chans=30,
        xdim=784,
        hdim=500,
        with_1b=True,
        with_2b=True,
    ):
        """ """
        self.with_1b, self.with_2b = with_1b, with_2b
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(784, zdim),
        )
        #
        self.dec_weight1 = torch.nn.Parameter(torch.randn(500, zdim, chans))
        self.dec_bias1 = torch.nn.Parameter(torch.randn(500, chans))
        #
        if with_1b:
            self.dec_weight1b = torch.nn.Parameter(
                torch.randn(500, chans, chans)
            )
            self.dec_bias1b = torch.nn.Parameter(torch.randn(500, chans))
        #
        self.dec_weight2 = torch.nn.Parameter(torch.randn(784, 500, chans))
        self.dec_bias2 = torch.nn.Parameter(torch.randn(784, chans))
        #
        if with_2b:
            self.dec_weight2b = torch.nn.Parameter(
                torch.randn(784, chans, chans)
            )
            self.dec_bias2b = torch.nn.Parameter(torch.randn(784, chans))
        #
        self.dec_weight3 = torch.nn.Parameter(torch.randn(784, chans))
        self.dec_bias3 = torch.nn.Parameter(torch.randn(784))
        #
        initialize_module(self, torch.nn.init.xavier_uniform_, 0)

    def forward(self, x):
        """ """
        x = self.enc(x)  # b, zdim
        #
        x = torch.einsum("oic,bi->boc", self.dec_weight1, x) + self.dec_bias1
        x = F.gelu(x)  # b, 500, c
        #
        if self.with_1b:
            x = torch.einsum("nck,bnc->bnk", self.dec_weight1b, x)
            x = F.gelu(x + self.dec_bias1b)
        #
        x = torch.einsum("oic,bic->boc", self.dec_weight2, x) + self.dec_bias2
        x = F.gelu(x)  # b, 784, c
        #
        if self.with_2b:
            x = torch.einsum("nck,bnc->bnk", self.dec_weight2b, x)
            x = F.gelu(x + self.dec_bias2b)
        #
        x = torch.einsum("nc,bnc->bn", self.dec_weight3, x) + self.dec_bias3
        x = F.sigmoid(x)
        return x, []


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE = torch.float32
    DEVICE = "cuda"  #  "cuda" if torch.cuda.is_available() else "cpu"
    FMNIST_PATH = os.path.join("datasets", "FashionMNIST")
    BATCH_SIZE = 60  # 30
    LR, MOMENTUM, WEIGHT_DECAY = 5e-4, 0, 0  #  1e-5, 0.9, 1e-4
    NUM_EPOCHS = 70

    train_ds = torchvision.datasets.FashionMNIST(
        FMNIST_PATH,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=True,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    model1 = MultichanMLP(350).to(DEVICE)  # RegularMLP(350).to(DEVICE)
    model2 = PCA_MLP(350).to(DEVICE)  # IdempotentMLP(350, rank=150).to(DEVICE)
    opt1 = torch.optim.Adam(
        model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    opt2 = torch.optim.Adam(
        model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    bce_fn = torch.nn.BCEWithLogitsLoss()
    l1_fn = torch.nn.L1Loss()
    mse_fn = torch.nn.MSELoss()
    loss_fn = mse_fn  # bce_fn
    #
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        for i, (imgs, targets) in enumerate(train_dl):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            opt1.zero_grad()
            opt2.zero_grad()
            #
            preds1, dists = model1(imgs.reshape(BATCH_SIZE, -1))
            preds1 = preds1.reshape(imgs.shape)
            # loss = bce_fn(preds1, imgs) + 0.1 * l1_fn(preds1, imgs)
            loss = loss_fn(preds1, imgs)
            loss.backward()
            opt1.step()
            if global_step % 200 == 0:
                print(
                    f"{epoch}/{i}",
                    "loss:",
                    loss.item(),
                    "dst:",
                    [d.item() for d in dists],
                )
            #
            preds2, dists = model2(imgs.reshape(BATCH_SIZE, -1))
            preds2 = preds2.reshape(imgs.shape)
            # loss = bce_fn(preds2, imgs)  # + 0.1 * l1_fn(preds2, imgs)
            loss = loss_fn(preds2, imgs)
            loss.backward()
            opt2.step()
            if global_step % 200 == 0:
                print(
                    "   [IDEMP]",
                    "loss:",
                    loss.item(),
                    "dst:",
                    [d.item() for d in dists],
                )
            #
            global_step += 1
    # 0.24 BCELoss is good
    # idx = 0; fig, (ax1, ax2, ax3) = plt.subplots(ncols=3); ax1.imshow(imgs[idx, 0].detach().cpu()), ax2.imshow(preds1[idx, 0].sigmoid().detach().cpu()); ax3.imshow(preds2[idx, 0].sigmoid().detach().cpu()); fig.show()
    # idx = 0; fig, (ax1, ax2, ax3) = plt.subplots(ncols=3); ax1.imshow(imgs[idx, 0].detach().cpu()), ax2.imshow(preds1[idx, 0].detach().cpu()); ax3.imshow(preds2[idx, 0].detach().cpu()); fig.show()
    # idx = 0; fig, (ax1, ax2) = plt.subplots(ncols=2); ax1.imshow(imgs[idx, 0].detach()), ax2.imshow(preds1[idx, 0].sigmoid().detach()); fig.show()
    breakpoint()
