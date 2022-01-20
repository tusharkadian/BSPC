# BSPC

Please cite the below paper if you use this repository-<br/>
[1] Atul Anand, Tushar Kadian, Manu Kumar Shetty, and Anubha Gupta, "Explainable AI Decision Model for ECG Data of Cardiac Disorders," Under Review, January 2022.

## Data Preprocessing for PTB-XL
1. Download the zip file from https://physionet.org/content/ptb-xl/1.0.1/
2. Extract to a location
3. Run the file <em>example_physionet.py</em>
4. It will initialize 4 variables: <em>X_train, y_train, X_test, y_test</em> 
5. Save the files in npy format: <code>np.save('<file_name>.npy', <variable_name>)</code>
6. These files could be used with our code.
  


## Instructions to run sample files
1. Clone the github repository to your local system using <code> git clone https://github.com/tusharkadian/BSPC.git </code>
2. Make sure python and pip are installed in your system. If not, visit https://www.python.org/ to download python. 
3. Code requires two more packages to run: **Numpy** and **Tensorflow**
   1. Install numpy -> <code> pip install numpy </code>
   2. Install tensorflow -> <code> pip install tensorflow </code>  
5. Run the code file: <code> python main.py </code>
6. It will output the classes for the given sample. 

Note: 
* We can specify the input file on line 12. 
* We can specify the model on line 18.
* Bilinear models required two stream of inputs. On line 22, pass <code>[x, x]</code> to model.predict().

## Trained models for PTB-XL ECG dataset

This repository contains our trained models which have been trained from scratch on PTB-XL ECG dataset (https://physionet.org/content/ptb-xl/1.0.1/).
It also contains models of other authors which we have trained on the same dataset after fine-tuning. If you intend to use any of the models in this repository, please cite the original authors reference as well as ours.

Please refer to the links below for the corresponding models adapted by us:
* ST-CNN-8: https://www.ahajournals.org/doi/full/10.1161/CIRCEP.119.007284
* ResNet based Models: https://github.com/raghakot/keras-resnet
* Attention-56 Model: https://github.com/Sourajit2110/Residual-Attention-Convolutional-Neural-Network
* SENet: https://ieeexplore.ieee.org/document/8578843

Our models, that are explained in the paper [1] referred above, include:
* ST-CNN-5
* ST-CNN-GAP-5
* DCT (H) + ST-CNN-GAP-5 
* DCT (HV) + ST-CNN-GAP-5 
* Bilinear ST-CNN-5 (Concatenate) 
* Bilinear ST-CNN-5 (Multiply) 
* Bilinear ST-CNN-5 (Outer Product)
