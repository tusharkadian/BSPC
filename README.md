# BSPC
## Trained models for PTB-XL ECG dataset

This repository contains our trained models which have been trained from scratch on PTB-XL ECG dataset (https://physionet.org/content/ptb-xl/1.0.1/).
It also contains models of other authors which we have trained on the same dataset after fine-tuning. If you intend to use any of the models in this repository, please cite the original authors reference as well as ours.

Please refer to the links below for the corresponding models adapted by us:
* ST-CNN-8: https://www.ahajournals.org/doi/full/10.1161/CIRCEP.119.007284
* ResNet based Models: https://github.com/raghakot/keras-resnet
* Attention-56 Model: https://github.com/Sourajit2110/Residual-Attention-Convolutional-Neural-Network
* SENet: https://ieeexplore.ieee.org/document/8578843

Our models include:
* ST-CNN-5
* ST-CNN-GAP-5
* DCT (H) + ST-CNN-GAP-5 
* DCT (HV) + ST-CNN-GAP-5 
* Bilinear ST-CNN-5 (Concatenate) 
* Bilinear ST-CNN-5 (Multiply) 
* Bilinear ST-CNN-5 (Outer Product)
