#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Loads dataset, prints stats, plots images with labels and metadata."""


import torch
import os
import json
import supervisely as sly
import matplotlib.pyplot as plt


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    PASCAL_PATH = os.path.join("datasets", "pascal_voc_2012")
    project = sly.Project(PASCAL_PATH, sly.OpenMode.READ)
    #
    print("Opened project: ", project.name)
    print("Number of images in project:", project.total_items)
    print("Number of datasets (folders) in project:", len(project.datasets))
    print(project.meta)
    #
    for obj_class in project.meta.obj_classes:
        print(
            f"Class '{obj_class.name}': geometry='{obj_class.geometry_type}',"
            f"color='{obj_class.color}'",
        )
    for tag in project.meta.tag_metas:
        print(f"Tag '{tag.name}': color='{tag.color}'")
    #
    for dataset in project.datasets:
        if dataset.name == "test":
            # no segmentation labels, skip
            continue
        for item_name, image_path, ann_path in dataset.items():
            print(f"Item '{item_name}': img='{image_path}', ann='{ann_path}'")
            img = sly.image.read(image_path)  # ndarray (h, w, 3) RGB
            #
            img_ann = img.copy()
            ann_json = json.load(open(ann_path))
            ann = sly.Annotation.from_json(ann_json, project.meta)
            label_details = ""
            for label in ann.labels:
                label_details += (
                    f"\n {label.obj_class.name}"
                    f" {label.obj_class.color}"
                    f" {label.obj_class.sly_id}"
                )

                label.draw(img_ann)
            #
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
            ax1.imshow(img)
            ax2.imshow(img_ann)
            fig.suptitle(image_path + label_details)
            fig.show()
            # Or alternatively draw annotation (all labels at once) preview with
            # ann.draw_pretty(img, output_path=res_image_path)
            breakpoint()
