# Idempotent DNN

----

# Getting started

### Dependencies

Create conda environment and install dependencies (can take a while):

```
conda create -n idemp python=3.10
conda activate idemp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install supervisely
pip install matplotlib
```


### Installing and testing PASCAL VOC:

Download PASCAL VOC 2012 segmentation dataset and decompress into `./datasets/pascal_voc_2012` (folders like `test`, `train`, `val` inside):

* Link: https://datasetninja.com/pascal-voc-2012

Dataset is downloaded in [supervisely format](https://developer.supervisely.com/getting-started/python-sdk-tutorials/common/iterate-over-a-local-project). The following script can be run to verify the installation. It should produce a plot with an image and its segmentations:

```
python 00a_verify_pascal.py
```

### Training a basic segmentation model:

The following script should initialize a baseline model and train it on image segmentation:

```
python 01a_train_baseline.py
```


----

# Research questions:

The scenario we seek is a DNN that achieves good performance with idempotent layers or blocks (i.e. if `y = layer(x)`, we also get `y == layer(y)`).

If this scenario is satisfied for the training data, we conjecture that at evaluation:
* If all layers retain idempotence, result can be trusted as in-distribution
* If at least one layer deviates from idempotence

Furthermore, we conjecture that:
* Enforcing idempotence may help with robustness/generalization
* It should be possible to create adversarial samples that break specifically one layer's idempotence
* Conversely, we can use the mismatch `y != layer(y)` to identify which properties of the sample are OOD

In summary, we would get an if-and-only-if OOD detection, with pullback-interpretability to the data space, at just the cost of 2 forwardpropagations.
