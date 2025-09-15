#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:

* we got an MLP that beats linear and almost overfits FMNIST with linear enc.
* Now, replace layers with idemp and obtain similar scenario
  - idemp encoder represents a subspace
  - ???
* Idempotence works, now leverage it for OOD. How??
* Move onto conv and segmentation

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

from idemp.models import LinearMLP, MultichannelMLP, initialize_module

from idemp.fftconv import circorr1d_fft, circorr2d_fft, CircularCorrelation
from idemp.fftconv import Circorr1d, Circorr2d
from idemp.projectors import (
    gaussian_projector,
    mutually_orthogonal_projectors,
    ProjStraightThrough,
)

from idemp.projectors import (
    IdempotentLinear,
    IdempotentCircorr1d,
    IdempotentCircorr2d,
)


# ##############################################################################
# # HELPERS
# ##############################################################################
class IdempotentEinsum(torch.nn.Module):
    """ """

    def __init__(
        self,
        wshape,
        wx_einstring,
        wb_einstring,
        rank,
        bias_shape=None,
        with_proj_dist=True,
        saturation=1.0,
    ):
        """ """
        super().__init__()
        self.wx_einstring = wx_einstring
        self.wb_einstring = wb_einstring
        self.rank = rank
        self.saturation = saturation
        self.weight = torch.nn.Parameter(torch.randn(*wshape))
        if bias_shape is not None:
            self.bias = torch.nn.Parameter(torch.zeros(*bias_shape))
            # self.bias.data.normal_(0.1)
        else:
            self.register_parameter("bias", None)
        self.with_proj_dist = with_proj_dist

    def forward(self, x):
        """ """
        w_proj = ProjStraightThrough.apply(
            self.weight.permute(2, 0, 1), self.rank, self.saturation
        ).permute(1, 2, 0)
        x = torch.einsum(self.wx_einstring, w_proj, x)
        if self.bias is not None:
            bias_negproj = self.bias - torch.einsum(
                self.wb_einstring, w_proj, self.bias
            )
            x = x + bias_negproj
        dist = torch.dist(self.weight, w_proj) if self.with_proj_dist else None
        return x, dist


class IdempotentMultichannelMLP(torch.nn.Module):
    """ """

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
        saturation=2.0,
    ):
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
        self.dec2 = IdempotentEinsum(
            (hdim, hdim, chans),
            "oic,bic->boc",
            "oic,ic->oc",
            rank=hdim // 2,
            bias_shape=(hdim, chans),
            with_proj_dist=True,
            saturation=saturation,
        )
        # self.dec_weight2 = torch.nn.Parameter(torch.randn(xdim, hdim, chans))
        # self.dec_bias2 = torch.nn.Parameter(torch.randn(xdim, chans))
        # #
        if with_2b:
            self.dec_weight2b = torch.nn.Parameter(
                torch.randn(hdim, chans, chans)
            )
            self.dec_bias2b = torch.nn.Parameter(torch.randn(hdim, chans))
        #
        self.dec3 = torch.nn.Linear(hdim * chans, xdim)
        # self.dec_weight3 = torch.nn.Parameter(torch.randn(xdim, chans))
        # self.dec_bias3 = torch.nn.Parameter(torch.randn(xdim))
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
        x, dst2 = self.dec2(x)
        x = self.activation(x)  # b, xdim, c
        #
        if self.with_2b:
            x = torch.einsum("nck,bnc->bnk", self.dec_weight2b, x)
            x = self.activation(x + self.dec_bias2b)  # b, xdim, c
        #
        x = self.dec3(x.view(len(x), -1))
        if not self.logits:
            x = F.sigmoid(x)  # b, xdim
        return x


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE = torch.float32
    DEVICE = "cuda"  #  "cuda" if torch.cuda.is_available() else "cpu"
    FMNIST_PATH = os.path.join("datasets", "FashionMNIST")
    BATCH_SIZE = 10  # 100
    LR, MOMENTUM, WEIGHT_DECAY = 5e-4, 0, 0  #  1e-5, 0.9, 1e-4
    NUM_EPOCHS = 3
    LOG_EVERY = 20

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

    model1 = MultichannelMLP(zdim=50, chans=50, hdim=350, logits=False).to(
        DEVICE
    )
    # model2 = LinearMLP(50, logits=False).to(DEVICE)
    model2 = IdempotentMultichannelMLP(
        zdim=50, chans=50, hdim=350, logits=False, saturation=2.0
    ).to(DEVICE)

    opt1 = torch.optim.Adam(
        model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    opt2 = torch.optim.Adam(
        model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    bce_fn = torch.nn.BCEWithLogitsLoss()
    l1_fn = torch.nn.L1Loss()
    mse_fn = torch.nn.MSELoss()
    loss_fn = mse_fn
    # loss_fn = lambda x, y: mse_fn(x, y) + 0.05 * l1_fn(x, y)
    #
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        if global_step > 200:
            break
        for i, (imgs, targets) in enumerate(train_dl):
            if global_step > 200:
                break
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            opt1.zero_grad()
            opt2.zero_grad()
            #
            preds1 = model1(imgs.reshape(BATCH_SIZE, -1))
            dists = []
            preds1 = preds1.reshape(imgs.shape)
            # loss = bce_fn(preds1, imgs) + 0.1 * l1_fn(preds1, imgs)
            loss = loss_fn(preds1, imgs)
            loss.backward()
            opt1.step()
            if global_step % LOG_EVERY == 0:
                print(
                    f"{epoch}/{i}",
                    "loss:",
                    loss.item(),
                    "dst:",
                    [d.item() for d in dists],
                )
            #
            preds2 = model2(imgs.reshape(BATCH_SIZE, -1))
            preds2 = preds2.reshape(imgs.shape)
            # loss = bce_fn(preds2, imgs)  # + 0.1 * l1_fn(preds2, imgs)
            loss = loss_fn(preds2, imgs)
            loss.backward()
            opt2.step()
            if global_step % LOG_EVERY == 0:
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

    # mmm = model1; sum(p.numel() for p in mmm.parameters() if p.requires_grad)
