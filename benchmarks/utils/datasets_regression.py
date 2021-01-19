import sys
import os
sys.path.append('../..')
sys.path.append('../')
import numpy as np
import gpsig

from utils.load_tsfiles import load_from_tsfile_to_dataframe 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from utils.tslearn_scaler import TimeSeriesScalerMeanVariance

def load_dataset_regression(dataset_name, for_model='sig', normalize_data=False, normalize_output=False, add_time=False, max_len=None, val_split=None, return_min_len=False):
    
    # if test_split is not None it will instead return test_split % of the training data for testing

    data_path_train = './datasets/{}_TRAIN.ts'.format(dataset_name)
    data_path_test = './datasets/{}_TEST.ts'.format(dataset_name)

    if not os.path.exists(data_path_train) or not os.path.exists(data_path_test):
        raise ValueError('Please download the attached datasets and extract to the /benchmarks/datasets/ directory...')

    X_train, y_train = load_from_tsfile_to_dataframe(data_path_train)
    X_test, y_test = load_from_tsfile_to_dataframe(data_path_test)

    if dataset_name=='PPGDalia':
        X_train = [np.stack([x[0][::2],x[1],x[2],x[3]], axis=1) for x in X_train.values]
        X_test = [np.stack([x[0][::2],x[1],x[2],x[3]], axis=1) for x in X_test.values]
    else:
        X_train = [np.stack(x, axis=1) for x in X_train.values]
        X_test = [np.stack(x, axis=1) for x in X_test.values]

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
    
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, shuffle=True)
    num_val = len(X_val)
    num_train = len(X_train)
    
    # if normalize_data:
        # scaler = StandardScaler()
        # scaler.fit(np.concatenate(X_train, axis=0))
        # X_train = [scaler.transform(x) for x in X_train]
        # X_val = [scaler.transform(x) for x in X_val] if X_val is not None else None
        # X_test = [scaler.transform(x) for x in X_test]
    
    if normalize_output:
        scaler = StandardScaler()
        scaler.fit(y_train[:,None])
        y_train = scaler.transform(y_train[:,None])[:,0]
        y_val = scaler.transform(y_val[:,None])[:,0]
        y_test = scaler.transform(y_test[:,None])[:,0]

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
    

    if normalize_data: 
        scaler = TimeSeriesScalerMeanVariance()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    if return_min_len:
        return X_train, y_train, X_val, y_val, X_test, y_test, len_min
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
