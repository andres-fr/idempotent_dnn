#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" """


import os
import torch
import matplotlib.pyplot as plt

from idemp.datasets import PascalSegmentationDataset

# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    PASCAL_PATH = os.path.join("datasets", "pascal_voc_2012")
    ds = PascalSegmentationDataset(
        PASCAL_PATH,
        split="train",
        img_transform=None,
        mask_transform=None,
    )
    img, ann, paths = ds[3]
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax1.imshow(img[0])
    ax2.imshow(ann)
    fig.show()
    breakpoint()
