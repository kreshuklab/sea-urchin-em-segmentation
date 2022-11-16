# Sea Urchin EM Segmentation

This repository contains scripts for (semi-manual) segmentation of neurons in a large EM-Volume of a sea urchin.
There are different scripts for the parts of the project:

## Training

The folder `training` contains the scripts for training neural networks for segmenting the neuron boundaries in EM.
Here, we make use of the method published in [Matskevych et al.](https://www.frontiersin.org/articles/10.3389/fcomp.2022.805166/full).
It segemnts the neuron boundaries in three steps:
1. Initial segmentation of the boundaries with ilastik pixel classification.
2. Improving the boundary segmentation with an *Enhancer*, a network that was trained to improve the pixel classification predictions on data with segmentation groundtruth from www.cremi.org.
3. Training a final segmentation network that learns to segment boundaries from the EM data directly using the predictions from the Enhancer as target signal.

The scripts `train_precomputed.py` and `train_pseudolabels.py` are used to train different versions of the final semgentation network, the other scripts are used for data preprocessing and for checking intermediate results.

## Segmentation

The folder `segmentation` contains the scripts to segment neuron fragments in the full EM volume. It uses the method from [Pape et al.](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Pape_Solving_Large_Multicut_ICCV_2017_paper.html) based on the boundary predictions of the segmentation network. This segmentation is then converted into a [paintera](https://github.com/saalfeldlab/paintera) project, which enables manually assembling full neurons out of the neuron fragments.

The script `segment_full_volume.py` runs the segmentation workflow, `to_paintera.py` converts the result to the paintera format.

## Data extraction

The folder `tif_extraction` contains scripts to extract data to other data formats that can be loaded with Amira:
- `extract_highres_cutout.py` to extract a cutout from the full EM raw data to tif (at full resolution).
- `extract_lowres_volume.py` to extract the full EM raw data to tif (at a low resolution).
- `extract_meshes_painera.py` to extract reconstructed neurons from paintera as meshes in the ply format.

## Installation

You can set-up a conda environment with all necessary dependencies to run these scripts via:
```
conda env create -f environment.yaml
```

Note that you will need a powerful workstation with a gpu or access to a computer cluster with gpu nodes to run the network training and the segmentation workflow.
