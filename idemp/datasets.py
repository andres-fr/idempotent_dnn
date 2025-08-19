#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
"""


import torch
import json
import supervisely as sly


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

    def __init__(
        self,
        root,
        split=None,
        img_transform=None,
        mask_transform=None,
    ):
        """ """
        self.project = sly.Project(root, sly.OpenMode.READ)
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        #
        self.paths = []
        for dataset in self.project.datasets:
            if split is not None and dataset.name != split:
                continue
            for item_name, image_path, ann_path in dataset.items():
                self.paths.append((image_path, ann_path))

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
        if self.img_transform is not None:
            img = self.img_transform(img)
        # load annotation as (h, w) int tensor. bg is 0, classe
        ann_json = json.load(open(ann_path))
        ann = sly.Annotation.from_json(ann_json, self.project.meta)
        mask = torch.zeros(img[0].shape, dtype=self.MASK_DTYPE)
        for label in ann.labels:
            class_idx = self.CLASS_TO_IDX[label.obj_class.name]
            label.draw(mask, class_idx)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        #
        return img, mask, (img_path, ann_path)
