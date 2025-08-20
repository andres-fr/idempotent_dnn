#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
"""

import json

import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import supervisely as sly

from .utils import integer_noise


# ##############################################################################
# # PASCAL 2012 SEGMENTATION
# ##############################################################################
class PascalSegmentationDataset(torch.utils.data.Dataset):
    """ """

    MASK_DTYPE = torch.int64
    CLASS_TO_IDX = {
        "neutral": 0,
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20,
    }
    IDX_TO_CLASS = [
        "neutral",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root,
        split=None,
        reshape_crop_hw=None,
        reshape_crop_seed_fn=None,
    ):
        """ """
        self.project = sly.Project(root, sly.OpenMode.READ)
        self.reshape_crop_hw = reshape_crop_hw
        self.reshape_crop_seed_fn = reshape_crop_seed_fn
        #
        self.paths = []
        for dataset in self.project.datasets:
            if split is not None and dataset.name != split:
                continue
            for item_name, image_path, ann_path in dataset.items():
                self.paths.append((image_path, ann_path))
        #
        self.normalize = transforms.Normalize(
            mean=self.RGB_MEAN, std=self.RGB_STD
        )

    def __len__(self):
        """ """
        return len(self.paths)

    def __getitem__(self, idx):
        """ """
        # load metadata
        img_path, ann_path = self.paths[idx]
        # load image as (3, h, w) float tensor normalized in [0, 1]
        img = sly.image.read(img_path)  # ndarray (H, W, 3), RGB
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        # load annotation as (h, w) int tensor. bg is 0, classe
        ann_json = json.load(open(ann_path))
        ann = sly.Annotation.from_json(ann_json, self.project.meta)
        if self.reshape_crop_hw is not None:
            img, mask = self.reshape_and_crop(img, ann, self.reshape_crop_hw)
        else:
            raise NotImplementedError
            mask = torch.zeros(img[0].shape, dtype=self.MASK_DTYPE)
            for label in ann.labels:
                class_idx = self.CLASS_TO_IDX[label.obj_class.name]
                label.draw(mask, class_idx)
        #
        img = self.normalize(img)
        return img, mask, (img_path, ann_path)

    def reshape_and_crop(
        self,
        img,
        sly_ann,
        shape=(520, 520),
    ):
        """ """
        mask = torch.zeros(img[0].shape, dtype=self.MASK_DTYPE)
        for label in sly_ann.labels:
            class_idx = self.CLASS_TO_IDX[label.obj_class.name]
            label.draw(mask, class_idx)
        #
        seed = (
            None
            if self.reshape_crop_seed_fn is None
            else self.reshape_crop_seed_fn()
        )
        out_h, out_w = shape
        _, h, w = img.shape
        ratio_h, ratio_w = out_h / h, out_w / w
        resize_ratio = max(ratio_h, ratio_w)
        if resize_ratio > 1:
            rsz = int(min(h, w) * resize_ratio + 1)
            img = F.resize(
                img, rsz, interpolation=F.InterpolationMode.BILINEAR
            )
            mask = F.resize(
                mask.unsqueeze(0),
                rsz,
                interpolation=F.InterpolationMode.NEAREST,
            ).squeeze(0)
        #
        mask_h, mask_w = mask.shape
        slack_h, slack_w = (mask_h - out_h), (mask_w - out_w)
        beg_h = integer_noise((1,), lo=0, hi=slack_h + 1, seed=seed).item()
        beg_w = integer_noise((1,), lo=0, hi=slack_w + 1, seed=seed).item()
        img = img[:, beg_h : beg_h + out_h, beg_w : beg_w + out_w]
        mask = mask[beg_h : beg_h + out_h, beg_w : beg_w + out_w]
        #
        return img, mask
