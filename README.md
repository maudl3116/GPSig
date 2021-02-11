# SigGPDE
Library for Gaussian process on sequential data using signature kernels as covariance functions, based on GPflow and TensorFlow.
Extends the GPSig library by:
- adding a new signature kernel, computed by solving a PDE
- adding a new sparse variational inference method based on variational orthogonal signature features. 
***
## Installing
Create and activate virtual environment with Python <= 3.7
```
conda create -n env_name python=3.7
conda activate env_name
```
Then, install the requirements using pip by
```
pip install -r requirements.txt
```
In order to use a GPU to run computations the following steps are required:
- Install a GPU compatible version of TensorFlow instead with the command
```
conda install -c anaconda tensorflow-gpu=1.15.3
pip install gpflow==1.5.1
```
- 
***
## Getting started
To get started, we suggest to first look at the notebook `signature_kernel.ipynb`, which gives a simple worked out example of how to use the signature kernel as a standalone object. In this notebook, we validate the implementation of the signature kernel by comparing our results to an alternative way of computing signature features using the `esig` package.
The difference between the two ways of computing the signature kernel is a 'kernel trick', which makes it possible to compute the signature kernel using only inner product evaluation on the underlying state-space.

In the other notebook, `ts_classification.ipynb`, a worked out example is given on how to use signature kernels for time series classification using inter-domain sparse variational inference with inducing tensors to make computations tractable and efficient. To make the most of these examples, we also recommend to look into the [GPflow](https://github.com/GPflow/GPflow) syntax of defining kernels and GP models, a Gaussian process library that we build on.
***

## Download datasets
The benchmarks directory contains the appropriate scripts used to run the benchmarking experiments in the paper. The datasets can be downloaded from our dropbox folder using the `download_data.sh` script in the `./benchmarks/datasets` folder by running
```
cd benchmarks
bash ./datasets/download_data.sh
```
or manually by copy-pasting the dropbox url containd within the aforementioned script.
