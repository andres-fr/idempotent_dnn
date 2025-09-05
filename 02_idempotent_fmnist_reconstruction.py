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
            raise NotImplementedError(f"Unknown weight: {pname}")


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


class RegularMLP(torch.nn.Module):
    """ """

    def __init__(self, hidden_dims=350, activation=torch.nn.LeakyReLU):
        """ """
        super().__init__()
        self.down = torch.nn.Linear(784, hidden_dims)
        self.body = torch.nn.Sequential(
            activation(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            activation(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            activation(),
            # torch.nn.Linear(hidden_dims, hidden_dims),
            # activation(),
        )
        self.up = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims, 784 * 3),
            activation(),
            torch.nn.Linear(784 * 3, 784 * 2),
            activation(),
            torch.nn.Linear(784 * 2, 784),
        )
        initialize_module(self, torch.nn.init.xavier_uniform_, 0)

    def forward(self, x):
        """ """
        dsts = []
        x = self.down(x)
        x = self.body(x)
        x = self.up(x)
        return x, dsts


class IdempotentMLP(torch.nn.Module):
    """
    This one works well on FMNIST trained for 15 epochs with BCELoss and
    Adam(lr=1e-3, wd=1e-5). 0.25 loss is already decent
    """

    def __init__(self, hidden_dims=350, rank=150, activation=torch.nn.Tanh):
        """ """
        super().__init__()
        self.down = torch.nn.Linear(784, hidden_dims)
        self.h1 = IdempotentLinear(hidden_dims, rank, bias=True, saturation=2)
        self.a1 = activation()
        self.up = torch.nn.Linear(hidden_dims, 784)

    def forward(self, x):
        """ """
        dsts = []
        x = self.down(x)
        x, dst1 = self.h1(x)
        dsts.append(dst1)
        x = self.a1(x)
        x = self.up(x)
        return x, dsts


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE = torch.float32
    DEVICE = "cuda"  #  "cuda" if torch.cuda.is_available() else "cpu"
    FMNIST_PATH = os.path.join("datasets", "FashionMNIST")
    BATCH_SIZE = 150
    LR, MOMENTUM, WEIGHT_DECAY = 1e-4, 0, 0  #  1e-5, 0.9, 1e-4
    NUM_EPOCHS = 50

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

    model1 = RegularMLP(350).to(DEVICE)
    model2 = PCA_MLP(350).to(DEVICE)  # IdempotentMLP(350, rank=150).to(DEVICE)
    opt1 = torch.optim.Adam(
        model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    opt2 = torch.optim.Adam(
        model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
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
    # idx = 0; fig, (ax1, ax2) = plt.subplots(ncols=2); ax1.imshow(imgs[idx, 0].detach()), ax2.imshow(preds1[idx, 0].sigmoid().detach()); fig.show()
    breakpoint()
