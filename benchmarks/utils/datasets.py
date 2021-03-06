import sys
import os
sys.path.append('../..')
sys.path.append('../')
import numpy as np
import gpsig
import pandas as pd
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
# from tslearn.datasets import UCR_UEA_datasets
# from sktime.utils.load_data import load_from_arff_to_dataframe
from utils.load_arff_files import load_from_arff_to_dataframe 


# for datasets that require a Fourier transform as preprocessing
from scipy import signal
import copy
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array
from utils.tslearn_scaler import TimeSeriesScalerMeanVariance

def load_dataset(dataset_name, for_model='sig', normalize_data=False, add_time=False, lead_lag=False, max_len=None, val_split=None, test_split=None, return_min_len=False):
    
    # if test_split is not None it will instead return test_split % of the training data for testing

    if dataset_name=='Crops':
        if not os.path.exists('./datasets/crops.csv'):
            raise ValueError('Please download the crops dataset') 
        data = pd.read_csv('./datasets/crops.csv',skiprows=1,header=None,encoding= 'unicode_escape')
        data = data.dropna()
        data = data[data[0]!='PK\x07\x08\x88<mßzW±\x01']
        data = data.values
  
        y, X = data[:,0].astype(int), data[:,1:][:,:,None]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y,random_state=0)

        if lead_lag:
            X_train = LeadLag().fit_transform(X_train) 
            X_test = LeadLag().fit_transform(X_test) 
    
    else:
        data_path = './datasets/{}.mat'.format(dataset_name)
   
        if not os.path.exists(data_path):
            data_path_train = './datasets/{0}/{0}_TRAIN.arff'.format(dataset_name)
            data_path_test = './datasets/{0}/{0}_TEST.arff'.format(dataset_name)

            if not os.path.exists(data_path_train):
                raise ValueError('Please download the attached datasets and extract to the /benchmarks/datasets/ directory...') 

            X_train, y_train = load_from_arff_to_dataframe('./datasets/{0}/{0}_TRAIN.arff'.format(dataset_name))
            X_test, y_test = load_from_arff_to_dataframe('./datasets/{0}/{0}_TEST.arff'.format(dataset_name))
            X_train = [np.stack(x, axis=1) for x in X_train.values]
            X_test = [np.stack(x, axis=1) for x in X_test.values]

            if dataset_name == 'RightWhaleCalls':
                X_train = np.array(spectrogram().fit_transform(X_train))
                X_test = np.array(spectrogram().fit_transform(X_test)) 
                labels_dict = {c : i for i, c in enumerate(np.unique(y_train))}
                y_train = np.asarray([labels_dict[c] for c in y_train])
                y_test = np.asarray([labels_dict[c] for c in y_test])
                
                scaler = TimeSeriesScalerMeanVariance()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
        else:
            data = loadmat(data_path)
            X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
            X_train, y_train, X_test, y_test = np.squeeze(X_train), np.squeeze(y_train), np.squeeze(X_test), np.squeeze(y_test)
            #X_train, y_train, X_test, y_test = UCR_UEA_datasets(use_cache=True).load_dataset(dataset_name)


    len_min = min(np.min([x.shape[0] for x in X_train]), np.min([x.shape[0] for x in X_test]))
    
    num_train = len(X_train)
    num_test = len(X_test)
    
    num_features = X_train[0].shape[1]
        
    if add_time:
        X_train = gpsig.preprocessing.add_time_to_list(X_train)
        X_test = gpsig.preprocessing.add_time_to_list(X_test)        
        num_features += 1
        
    if max_len is not None:
        # perform mean-pooling of every n subsequent observations such that the length of each sequence <= max_len
        X_train = [x if x.shape[0] <= max_len else
                    np.stack([x[i*int(np.ceil(x.shape[0]/max_len)):np.minimum((i+1)*int(np.ceil(x.shape[0]/max_len)), x.shape[0])].mean(axis=0)
                                for i in range(int(np.ceil(x.shape[0]/np.ceil(x.shape[0]/max_len))))], axis=0) for x in X_train]
        X_test = [x if x.shape[0] <= max_len else
                    np.stack([x[i*int(np.ceil(x.shape[0]/max_len)):np.minimum((i+1)*int(np.ceil(x.shape[0]/max_len)), x.shape[0])].mean(axis=0)
                            for i in range(int(np.ceil(x.shape[0]/np.ceil(x.shape[0]/max_len))))], axis=0) for x in X_test]

    num_classes = np.unique(np.int32(y_train)).size
    
    if val_split is not None:
        if val_split < 1. and np.ceil(val_split * num_train) < 2 * num_classes:
            val_split = 2 * num_classes
        elif val_split > 1. and val_split < 2 * num_classes:
            val_split = 2 * num_classes
    
    if test_split is not None:
        if test_split < 1. and np.ceil(test_split * num_train) < 2 * num_classes:
            test_split = 2 * num_classes
        elif test_split > 1. and test_split < 2 * num_classes:
            test_split = 2 * num_classes
    
    if val_split is not None and test_split is not None:
        if val_split < 1. and test_split > 1:
            val_split = int(np.ceil(num_train * val_split))
        elif val_split > 1 and test_split < 1.:
            test_split = int(np.ceil(num_train * test_split))
                
    split_from_train = val_split + test_split if val_split is not None and test_split is not None else val_split or test_split 

    if split_from_train is not None:

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_from_train, shuffle=True, stratify=y_train)
        
        if val_split is not None and test_split is not None:
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=float(test_split)/split_from_train, shuffle=True, stratify=y_val)
            num_val = len(X_val)
            num_test = len(X_test)
        elif val_split is not None:
            num_val = len(X_val)
        else:
            X_test, y_test = X_val, y_val
            X_val, y_val = None, None
            num_test = len(X_test)
            num_val = 0
        num_train = len(X_train)
    else:
        X_val, y_val = None, None
        num_val = 0

    if normalize_data:
        scaler = StandardScaler()
        scaler.fit(np.concatenate(X_train, axis=0))
        X_train = [scaler.transform(x) for x in X_train]
        X_val = [scaler.transform(x) for x in X_val] if X_val is not None else None
        X_test = [scaler.transform(x) for x in X_test]
    
    for_model = for_model.lower()
    if X_val is None:
        if for_model.lower() == 'sig':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_test)
        elif for_model.lower() == 'nn':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_test, pre=True, pad_with=0.)
        elif for_model.lower() == 'kconv':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_test, pad_with=float('nan'))
        else:
            raise ValueError('unknown architecture: {}'.format(for_model))
        X_train = X[:num_train]
        X_test = X[num_train:]
    else:
        if for_model.lower() == 'sig':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_val + X_test)
        elif for_model.lower() == 'nn':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_val + X_test, pre=True, pad_with=0.)
        elif for_model.lower() == 'kconv':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_val + X_test, pad_with=float('nan'))
        else:
            raise ValueError('unknown architecture: {}'.format(for_model))
        X_train = X[:num_train]
        X_val = X[num_train:num_train+num_val]
        X_test = X[num_train+num_val:]
    
    labels = {y : i for i, y in enumerate(np.unique(y_train))}

    y_train = np.asarray([labels[y] for y in y_train])
    y_val = np.asarray([labels[y] for y in y_val]) if y_val is not None else None
    y_test = np.asarray([labels[y] for y in y_test])
    
    if return_min_len:
        return X_train, y_train, X_val, y_val, X_test, y_test, len_min
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test


# for the Whale dataset
class spectrogram(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        frequencies, times, spectrogram = signal.spectrogram(X,fs=4000,nfft=256,noverlap=128)
        # spectrogram = scipy.ndimage.filters.gaussian_filter(spectrogram, [1.1,1.1], mode='constant')
        return spectrogram.T[:,2:30]
        

    def transform(self, X, y=None):
        return [self.transform_instance(x[:,0]) for x in X]

class LeadLag(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        lag = []
        lead = []

        for val_lag, val_lead in zip(X[:-1], X[1:]):
            lag.append(val_lag)
            lead.append(val_lag)

            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(X[-1])
        lead.append(X[-1])

        return np.c_[lag, lead]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]