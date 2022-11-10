# Adaptive Multiscale Superpixel Embedding Convolutional Neural Network for Land Use Classification

This repository is the official implementation of [Adaptive Multiscale Superpixel Embedding Convolutional Neural Network for Land Use Classification](https://ieeexplore.ieee.org/document/9875975).

Code to build and train the networks are provided. The model takes in a provided remote sensing image and produces a colour coded output depending on provided categories.

> :warning: 
**This code currently has an undetermined memory issue as a result of incompatibility between older code library versions**: 

It is not advised to run this code right now and any responsibility will fall onto the user for subsequent issues if used!

> :warning:


![AMUSE-CNN Architecture](images/amuse_cnn.jpg "AMUSE CNN Architecture")

## Published Results


## Paper

[IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing](https://ieeexplore.ieee.org/document/9875975)

## Abstract

Currently, a large number of remote sensing images with different resolutions are available for Earth observation and land monitoring, which are inevitably demanding intelligent analysis techniques for accurately identifying and classifying land use (LU). This article proposes an adaptive multiscale superpixel embedding convolutional neural network architecture (AMUSE-CNN) for tackling LU classification. Initially, the images are parsed via the superpixel representation so that the object-based analysis (via a superpixel embedding convolutional neural network scheme) can be carried out with the pixel context and neighborhood information. Then, a multiscale convolutional neural network (MS-CNN) is proposed to classify the superpixel-based images by identifying object features across a variety of scales simultaneously, in which multiple window sizes are used to fit to the various geometries of different LU classes. Furthermore, a proposed adaptive strategy is applied to best exert the classification capability of the MS-CNN. Subsequently, two modules are developed to fully implement the AMUSE-CNN architecture. More specifically, Module I is to determine the most suitable classes for each window size (scale) by applying majority voting to a series of MS-CNNs Module II carries out the classification of the classes identified in Module I for the given scale used in the MS-CNN and, therefore, complete the LU classification of the entire classes. The proposed AMUSE-CNN architecture is both quantitatively and qualitatively validated using remote sensing data collected from two cities, Kano and Lagos in Nigeria, due to the spatially complex LU distribution. Experimental results show the superior performance of our approach against several state-of-the-art techniques.

## Requirements

### Python and Operating System

The following code was written in Python 3.8 and run in Windows 10.

The above requirements list contains all packages that <b>must</b> be installed in order for the code to run correctly.

To install these packages, either copy the above into a text file, or download the requirements.txt file within the repository and install using pip as shown below

```setup
pip install -r requirements.txt
```

```
rtools==9.0.9048
rasterio~=1.1.7
numpy~=1.19.4
shapely~=1.7.1
affine~=2.3.0
scikit-learn~=0.23.2
tensorflow~=2.4.1
tqdm~=4.50.2
fiona~=1.8.17
geopandas~=0.8.1
scipy~=1.5.2
matplotlib~=3.3.2
pillow~=7.2.0
scikit-image~=0.17.2
```

Note: While operations can be performed using a CPU based tensorflow installation, it will be extremely slow and therefore a GPU based installation is recommended.


Tensorflow will require additional support to make use of GPU support. Details of these required elements can be found at

[Tensorflow GPU Support](https://tensorflow.org/install/gpu)