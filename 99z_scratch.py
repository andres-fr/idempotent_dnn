# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    DTYPE = torch.float32
    DEVICE = "cpu"  #  "cuda" if torch.cuda.is_available() else "cpu"
    FMNIST_PATH = os.path.join("datasets", "FashionMNIST")
    BATCH_SIZE = 40
    LR, MOMENTUM, WEIGHT_DECAY = 1e-4, 0, 1e-6  #  1e-5, 0.9, 1e-4

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
