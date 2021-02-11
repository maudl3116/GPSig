# SigGPDE
A repository for Gaussian Processes on sequential data using signature kernels as covariance functions.
SigGPDE is built upon the library [GPSig](https://arxiv.org/abs/1906.08215) which is itself based on GPflow and TensorFlow. 
The new features in SigGPDE are:
- computation of the signature kernels using the [PDE-based kernel trick](https://arxiv.org/pdf/2006.14794.pdf)
- computation of the gradients of the signature kernels without resorting to automatic differentation
- sparse variational inference method based on variational orthogonal signature features
***
## Installing
Create and activate virtual environment with Python <= 3.7
```
conda create -n sigGPDE_env python=3.7
conda activate sigGPDE_env
```
Then, install the requirements using pip by
```
pip install -r requirements.txt
```
Note that in order to use a GPU to run computations you need to install a GPU compatible version of TensorFlow as follows
```
conda install -c conda-forge tensorflow-gpu=1.15
pip install gpflow==1.5.1
```
### Building the signature kernel operator (GPU)
Build the custom TensorFlow operator as follows
```
cd gpsig/covariance_op
sh Makefile_gpu.sh
```
You may have to modify the Makefile with your own cuda and tensorflow paths.  

### Building the signature kernel operator (CPU)
If you do not have a GPU, you need to build the Cython operator by executing
```
cd gpsig
python setup.py build_ext --inplace
```
***
## Source code
The main functionalities (in the folder `gpsig`) for SigGPDE can be found in:
- `kernels_pde.py` where the PDE signature kernel is implemented
- `inducing_variables_vosf.py` where the variational orthogonal inducing signature features are implemented
## Notebooks
### PDE Signature kernel
The notebook `pde_signature_kernel.ipynb` shows how to use the pde signature kernel. In this notebook, we validate the implementation of the PDE signature kernel and its gradients by comparing our results to the signature kernel trick used in GPSig. The notebook can also be used to verify that you have successfully built the signature kernel operators (Cython or Cuda). 
### Classification of time series with SigGPDE
The notebook `classification_example.ipynb` shows how to use SigGPDE for time series classification.
### Forecasting rainfall with SigGPDE
The notebook `rainfall_forecast.ipynb` shows how SigGPDE can be used to predict whether it will rain in the next hour using historical climatic data.
***

## Benchmarks and Datasets
The benchmarks directory contains the appropriate scripts used to run the benchmarking experiments in the paper. The datasets can be downloaded using the `download_data.sh` script in the `./benchmarks/datasets` folder by running
```
cd benchmarks
sh ./datasets/download_data.sh
```
