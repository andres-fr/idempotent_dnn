#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TODO:
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
class IdempotentMLP(torch.nn.Module):
    """ """

    def __init__(
        self,
        in_dims=784,
        hidden_dims=100,
        hidden_rank=20,
        num_hidden=3,
        saturation=2.0,
    ):
        """ """
        super().__init__()
        self.head = torch.nn.Linear(in_dims, hidden_dims, bias=True)
        # self.body = [
        #     IdempotentLinear(hidden_dims, hidden_rank, bias=True, saturation=saturation)
        #     for _ in range(num_hidden)
        # ]
        self.test1 = IdempotentLinear(
            hidden_dims, hidden_rank, bias=True, saturation=saturation
        )
        self.test2 = IdempotentLinear(
            hidden_dims, hidden_rank, bias=True, saturation=saturation
        )
        self.tail = torch.nn.Linear(hidden_dims, in_dims, bias=True)

    def forward(self, x):
        """ """
        x = self.head(x)
        dists = []
        # for lyr in self.body:
        #     x, dist = lyr(x)
        #     dists.append(dist)
        # x, dist1 = self.test1(x)
        # dists.append(dist1)
        # x, dist2 = self.test2(x)
        # dists.append(dist2)
        x = torch.tanh(x)
        x = self.tail(x)
        return x, dists


class MNIST_Net_Baseline(torch.nn.Module):
    """
    This one works well on FMNIST trained for 10 epochs with BCELoss and
    Adam(lr=1e-4).
    """

    def __init__(self, activation=torch.nn.Tanh):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 784),
            activation(),
            torch.nn.Linear(784, 512),
            activation(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 784),
            activation(),
            torch.nn.Linear(784, 784),
            torch.nn.Sigmoid(),  # Sigmoid to get output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, []


class MNIST_Net(torch.nn.Module):
    """
    This one works well on FMNIST trained for 10 epochs with BCELoss and
    Adam(lr=1e-4). 0.24 loss is already decent
    """

    def __init__(self, activation=torch.nn.Tanh):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            activation(),
            # torch.nn.Linear(512, 512),
            # activation(),
            # torch.nn.Linear(512, 512),
        )
        self.ip1 = IdempotentLinear(512, 500, bias=True, saturation=2.0)
        self.ip2 = IdempotentLinear(512, 500, bias=True, saturation=2.0)
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(512, 512),
            # activation(),
            # torch.nn.Linear(512, 512),
            # activation(),
            torch.nn.Linear(512, 784),
            torch.nn.Sigmoid(),  # Sigmoid to get output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x, dst1 = self.ip1(x)
        x, dst2 = self.ip2(x)
        x = self.decoder(x)
        return x, [dst1, dst2]


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE = torch.float32
    DEVICE = "cpu"  #  "cuda" if torch.cuda.is_available() else "cpu"
    FMNIST_PATH = os.path.join("datasets", "FashionMNIST")
    BATCH_SIZE = 40
    LR, MOMENTUM, WEIGHT_DECAY = 1e-4, 0, 1e-6  #  1e-5, 0.9, 1e-4

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

    # model = IdempotentMLP(
    #     in_dims=784, hidden_dims=700, hidden_rank=650, num_hidden=1
    # ).to(DEVICE)
    model = MNIST_Net().to(DEVICE)
    # opt = torch.optim.SGD(
    #     model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    # )
    opt = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.BCELoss()
    #
    global_step = 0
    for epoch in range(15):
        for i, (imgs, targets) in enumerate(train_dl):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            opt.zero_grad()
            preds, dists = model(imgs.reshape(BATCH_SIZE, -1))
            preds = preds.reshape(imgs.shape)
            loss = loss_fn(preds, imgs)
            loss.backward()
            opt.step()
            if global_step % 200 == 0:
                print(
                    f"{epoch}/{i}",
                    "loss:",
                    loss.item(),
                    "dst:",
                    [d.item() for d in dists],
                )
            global_step += 1

    # 0.24 BCELoss is good
    # idx = 0; fig, (ax1, ax2) = plt.subplots(ncols=2); ax1.imshow(imgs[idx, 0].detach()), ax2.imshow(preds[idx, 0].detach()); fig.show()
    breakpoint()
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

    num_blocks, in_chans, out_chans = 3, 11, 5
    blocks = list(
        # gaussian_projector(
        #     chans,
        #     max(1, chans // 2),
        #     orth=True,
        #     seed=10000 + i,
        #     dtype=DTYPE,
        #     device=DEVICE,
        # )[0]
        gaussian_noise(
            (out_chans, in_chans), seed=10000 + i, dtype=DTYPE, device=DEVICE
        )
        for i in range(num_blocks)
    )
    kernel = (
        torch.hstack(blocks)
        .reshape(out_chans, num_blocks, in_chans)
        .permute(0, 2, 1)
    )
    x = gaussian_noise(
        (in_chans, num_blocks), seed=12345, dtype=DTYPE, device=DEVICE
    )
    y1 = CircularCorrelation.circorr1d(x, kernel)
    y2 = CircularCorrelation.circorr1d_fft(x, kernel).real

    # cc1d = Circorr1d(11, 5, 3, bias=False)
    # cc1d.kernel.data[:] = kernel
    # cc1d(x.unsqueeze(0))
    quack = IdempotentCircorr1d(11, 3, 5, bias=True)
    yy1, _ = quack(x.unsqueeze(0))
    yy2, _ = quack(yy1)

    #
    #
    #

    hh, ww = (13, 17)

    kernel2d = gaussian_noise(
        (out_chans, in_chans, hh, ww), seed=1234, dtype=DTYPE, device=DEVICE
    )
    xx = gaussian_noise(
        (in_chans, hh, ww), seed=1205, dtype=DTYPE, device=DEVICE
    )
    yy1 = CircularCorrelation.circorr2d(xx, kernel2d)
    yy2 = CircularCorrelation.circorr2d_fft(xx, kernel2d)

    #
    #
    #
    quack = IdempotentCircorr2d(11, (3, 3), 5, bias=True)
    yyy1, dst1 = quack(xx.unsqueeze(0))
    yyy2, dst2 = quack(yyy1)

    # in this case we observe that our conv1d indeed performs dotprods as-is,
    # and shifts the kernel across the signal. x=(b, in, n), k=(out,in,n)
    x = torch.arange(10, dtype=DTYPE, device=DEVICE).reshape(2, 1, 5)
    k = torch.zeros((1, 1, 5), dtype=DTYPE, device=DEVICE)
    k[0, 0, 0] = 1
    # k[0, 0, 1] = 1
    y = circorr1d_fft(x, k).real

    # in the 2d case, same
    xx = (
        torch.arange(25, dtype=DTYPE, device=DEVICE).reshape(5, 5).unsqueeze(0)
    )
    xx = torch.stack([xx, xx * 2])
    kk = torch.zeros((1, 1, 5, 5), dtype=DTYPE, device=DEVICE)
    kk[0, 0, 2, 2] = 1
    yy = circorr2d_fft(xx, kk).real

    CC1 = Circorr1d(1, 1, 3, bias=True)

    loss_fn = torch.nn.MSELoss()

    #
    #
    xx = (  # (1, 11, 5, 5)  bchw
        torch.arange(25, dtype=DTYPE, device=DEVICE)
        .view(1, 1, -1)
        .reshape(1, 1, 5, 5)
        .repeat(1, 11, 1, 1)
    )
    CC2 = IdempotentCircorr2d(11, (3, 3), 5, bias=True)
    opt = torch.optim.SGD(
        CC2.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-2
    )
    for i in range(40000):
        opt.zero_grad()
        z, dst = CC2(xx)
        loss = loss_fn(z, xx)  # + 0.01 * dst
        loss.backward()
        opt.step()
        if i % 500 == 0:
            print(i, "loss:", loss.item(), "dst:", dst.item())

    breakpoint()

    # plt.clf(); plt.plot(CC2.bias.detach().numpy()); plt.show()
    # plt.clf(); plt.plot(CC2.bias.detach().numpy()); plt.show()
    #
    #
    #
    CC2 = Circorr2d(1, 1, (3, 4), bias=True)
    opt = torch.optim.SGD(
        CC2.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-2
    )
    for i in range(30000):
        opt.zero_grad()
        z = CC2(xx)
        loss = loss_fn(z, xx)
        loss.backward()
        opt.step()
        if i % 500 == 0:
            print(i, "loss:", loss)
    breakpoint()
