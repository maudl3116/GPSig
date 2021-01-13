import os
import sys

sys.path.append('../..')
sys.path.append('..')

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gpflow as gp
import gpsig

import pickle
import matplotlib.pyplot as plt

from utils import *

from sklearn.metrics import accuracy_score, classification_report
from gpsig.precompute_signatures import SignatureCalculator

def get_signatures(signature_calculator, data):
    return signature_calculator.compute_signature(data)

def train_gpsig_vosf_classifier(dataset, inf = True, sig_precompute=True, num_levels=5, M=500, normalize_data=True, minibatch_size=50, max_len=500,
                           num_lags=None, order =0, fast_algo = False, val_split=None, test_split=None, experiment_idx=None, save_dir='./GPSig/'):
    
    """
        # Inputs:
        ## Args:
        :inf:                  if True, then we use the GP model with untruncated signature kernel. 
        :num_levels            this has to be specified for the GP model with truncated signature kernel. It corresponds to the level of truncation of the latter.
        :M                     the number of inducing variables 
        :order                 corresponds to level of discretization for the untruncated signature kernel
    """

    print('####################################')
    print('Training dataset: {}'.format(dataset))
    print('####################################')
    if sig_precompute:
        compute_sig=False
    else:
        compute_sig=True
        
    if fast_algo:
        assert sig_precompute==False, "no need to precompute signatures if using the fast algorithm"
        qdiag = True
    else:
        # assert sig_precompute==True, "should precompute the signatures"
        qdiag = False
    
    ## load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset, val_split=val_split, test_split=test_split,
                                                                  normalize_data=normalize_data, add_time=True, for_model='sig', max_len=max_len)
    
    # when precomputing the signatures, we need to compute the truncation level which ensures that we can compute the M inducing variables
    n_M = gpsig.utils.compute_trunc(M,X_train.shape[2])
    if not inf:
        assert n_M <= num_levels, "Does not make sense to use n_M > num_levels"

    ## need to add possibiliy of rescaling here. 
    if sig_precompute: 
        signature_train_calculator = SignatureCalculator(dataset=dataset, dataset_type='train',
                                                   truncation_level=n_M, add_time=True)
        signature_val_calculator = SignatureCalculator(dataset=dataset, dataset_type='val',
                                                   truncation_level=n_M, add_time=True) 
        signature_test_calculator = SignatureCalculator(dataset=dataset, dataset_type='test',
                                                   truncation_level=n_M, add_time=True)


        S_train = get_signatures(signature_train_calculator, X_train)
        S_val = get_signatures(signature_val_calculator, X_val)
        S_test = get_signatures(signature_test_calculator, X_test)

    num_train, len_examples, num_features = X_train.shape
    num_val = X_val.shape[0] if X_val is not None else None
    num_test = X_test.shape[0]
    num_classes = np.unique(y_train).size

    ## compute and print the number of inducing features (could also use S_train.shape[1]+1)
    # num_inducing = np.sum([1 for k in range(n_M+1) for repeat in range(num_features**k)])
    # print('number of inducing features: ', num_inducing)
    
    with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        
        ## initialize inducing tensors and lengthsacles        
        l_init = suggest_initial_lengthscales(X_train, num_samples=1000)

        ## reshape data into 2 axes format for gpflow
        input_dim = len_examples * num_features
        X_train = X_train.reshape([-1, input_dim])
        X_val = X_val.reshape([-1, input_dim]) if X_val is not None else None
        X_test = X_test.reshape([-1, input_dim])
        
        ## setup model
        if inf:
            feat = gpsig.inducing_variables_vosf.UntruncInducingOrthogonalTensors(input_dim=input_dim, d = num_features, M = M, num_lags=num_lags, compute_sig=compute_sig) 
        else:
            feat = gpsig.inducing_variables_vosf.TruncInducingOrthogonalTensors(input_dim=input_dim, d = num_features, M = M)

        ## define kernel
        #k = gpsig.kernels.SignatureRBF(input_dim, num_levels=num_levels, num_features=num_features, lengthscales=l_init, num_lags=num_lags, low_rank=low_rank)
        if inf:
            k = gpsig.kernels_pde.UntruncSignatureKernel(input_dim, num_features, order=order, lengthscales=l_init, num_lags=num_lags)
        else:
            k = gpsig.kernels.SignatureLinear(input_dim, num_features=num_features, num_levels=num_levels, order=num_levels, lengthscales=l_init,normalization=False, difference=True)
        
        if num_classes == 2:
            lik = gp.likelihoods.Bernoulli()
            num_latent = 1
        else:
            lik = gp.likelihoods.MultiClass(num_classes)
            num_latent = num_classes

        if sig_precompute:
            X_train = np.concatenate([X_train,S_train[:,:M-1]],axis=1)
            X_val = np.concatenate([X_val,S_val[:,:M-1]],axis=1)
            X_test = np.concatenate([X_test,S_test[:,:M-1]],axis=1)

        m = gpsig.models.SVGP(X_train, y_train[:, None], kern=k, feat=feat, likelihood=lik, num_latent=num_latent, q_diag = qdiag,
                              minibatch_size=minibatch_size if minibatch_size < num_train else None, whiten=True, fast_algo=fast_algo)

        ## setup metrics
        def batch_predict_y(m, X, batch_size=None):
            num_iters = int(np.ceil(X.shape[0] / batch_size))
            y_pred = np.zeros((X.shape[0]), dtype=np.float64)
            for i in range(num_iters):
                slice_batch = slice(i*batch_size, np.minimum((i+1)*batch_size, X.shape[0]))
                X_batch = X[slice_batch]
                pred_batch = m.predict_y(X_batch)[0]
                if pred_batch.shape[1] == 1:
                    y_pred[slice_batch] = pred_batch.flatten() > 0.5
                else:
                    y_pred[slice_batch] = np.argmax(pred_batch, axis=1)
            return y_pred

        def batch_predict_density(m, X, y, batch_size=None):
            num_iters = int(np.ceil(X.shape[0] / batch_size))
            y_nlpp = np.zeros((X.shape[0]), dtype=np.float64)
            for i in range(num_iters):
                slice_batch = slice(i*batch_size, np.minimum((i+1)*batch_size, X.shape[0]))
                X_batch = X[slice_batch]
                y_batch = y[slice_batch, None]
                y_nlpp[slice_batch] = m.predict_density(X_batch, y_batch).flatten()
            return y_nlpp

        acc = lambda m, X, y: accuracy_score(y, batch_predict_y(m, X, batch_size=minibatch_size))
        nlpp = lambda m, X, y: -np.mean(batch_predict_density(m, X, y, batch_size=minibatch_size))

        val_acc = lambda m: acc(m, X_val, y_val)
        val_nlpp = lambda m: nlpp(m, X_val, y_val)
        
        test_acc = lambda m: acc(m, X_test, y_test)
        test_nlpp = lambda m: nlpp(m, X_test, y_test)

        val_scorers = [val_acc, val_nlpp] if X_val is not None else None

        ## train model
        opt = gpsig.training.NadamOptimizer
        num_iter_per_epoch = int(np.ceil(float(num_train) / minibatch_size))
        
        ### phase 1 - pre-train variational distribution
        print_freq = 10 #np.minimum(num_iter_per_epoch, 100)
        save_freq = 100 #np.minimum(num_iter_per_epoch, 50)
        patience = np.maximum(500 * num_iter_per_epoch, 5000)
        
        m.kern.set_trainable(False)
        hist = gpsig.training.optimize(m, opt(1e-3), max_iter=patience, print_freq=print_freq, save_freq=save_freq,
                                       val_scorer=val_scorers, save_best_params=X_val is not None, lower_is_better=True)
        
        ### phase 2 - train kernel (with sigma_i=sigma_j fixed) with early stopping
        m.kern.set_trainable(True)
        # m.kern.variances.set_trainable(False)
        hist = gpsig.training.optimize(m, opt(1e-3), max_iter=patience, print_freq=print_freq, save_freq=save_freq, history=hist, # global_step=global_step,
                                       val_scorer=val_scorers, save_best_params=X_val is not None, lower_is_better=True, patience=patience)
        ### restore best parameters
        if 'best' in hist and 'params' in hist['best']: m.assign(hist['best']['params'])
                
        # ### phase 3 - train with all kernel hyperparameters unfixed
        # # m.kern.variances.set_trainable(True)
        # hist = gpsig.training.optimize(m, opt(1e-3), max_iter=5000*num_iter_per_epoch, print_freq=print_freq, save_freq=save_freq, history=hist, # global_step=global_step,
        #                               val_scorer=val_scorers, save_best_params=X_val is not None, lower_is_better=True, patience=patience)
        # ### restore best parameters
        # if 'best' in hist and 'params' in hist['best']: m.assign(hist['best']['params'])
        
        ### evaluate on validation data
        val_nlpp = val_nlpp(m)
        val_acc = val_acc(m)

        print('Val. nlpp: {:.4f}'.format(val_nlpp))
        print('Val. accuracy: {:.4f}'.format(val_acc))
            
        ### phase 4 - fix kernel parameters and train on rest of data to assimilate into variational approximation
        m.kern.set_trainable(False)
	
        if val_split is not None:
            X_train, y_train = np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0)
            m.X, m.Y = X_train, y_train
            num_train = X_train.shape[0]
            m.num_data = num_train
            
        hist = gpsig.training.optimize(m, opt(1e-3), max_iter=patience, print_freq=print_freq, save_freq=save_freq, history=hist)
        
        ## evaluate on test data
        test_nlpp = nlpp(m, X_test, y_test)
        test_acc = acc(m, X_test, y_test)
        test_report = classification_report(y_test, batch_predict_y(m, X_test, batch_size=minibatch_size))

        print('Test nlpp: {:.4f}'.format(test_nlpp))
        print('Test accuracy: {:.4f}'.format(test_acc))
        print(test_report)

        ## save results to file
        hist['results'] = {}
        hist['results']['val_acc'] = val_acc
        hist['results']['val_nlpp'] = val_nlpp
        hist['results']['test_nlpp'] = test_nlpp
        hist['results']['test_acc'] = test_acc
        hist['results']['test_report'] = test_report
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        experiment_name = '{}'.format(dataset)
        if experiment_idx is not None:
            experiment_name += '_{}'.format(experiment_idx)
        with open(os.path.join(save_dir, experiment_name + '.pkl'), 'wb') as f:
            pickle.dump(hist, f)
        with open(os.path.join(save_dir, experiment_name + '.txt'), 'w') as f:
            f.write('Val. nlpp: {:.4f}\n'.format(val_nlpp))
            f.write('Val. accuracy: {:.4f}\n'.format(val_acc))
            f.write('Test nlpp: {:.4f}\n'.format(test_nlpp))
            f.write('Test accuracy: {:.4f}\n'.format(test_acc))
            f.write('Test report:\n')
            f.write(test_report)
            
    ## clear memory manually
    gp.reset_default_session()
    tf.reset_default_graph()

    import gc
    gc.collect()

    return
