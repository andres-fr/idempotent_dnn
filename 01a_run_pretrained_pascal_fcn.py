#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
"""


import os
import torch
import matplotlib.pyplot as plt


from idemp.datasets import PascalSegmentationDataset

from torchvision.io.image import decode_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    PASCAL_PATH = os.path.join("datasets", "pascal_voc_2012")
    TRAIN_HW = (520, 520)
    BATCH_SIZE = 4
    # load dataset
    train_ds = PascalSegmentationDataset(
        PASCAL_PATH, split="train", reshape_crop_hw=TRAIN_HW
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    # load pretrained model
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    model.eval()
    #
    for imgs, masks, paths in train_dl:
        with torch.no_grad():
            preds = model(imgs)["out"].softmax(dim=1)
            pred_masks = preds.softmax(dim=1)
            quack = preds.argmax(dim=1)
            #
            fig, (ax1, ax2, ax3) = plt.subplots(
                ncols=3, sharex=True, sharey=True
            )
            ax1.imshow(imgs[0, 0])
            ax2.imshow(masks[0])
            ax3.imshow(quack[0])
            fig.show()
            breakpoint()
