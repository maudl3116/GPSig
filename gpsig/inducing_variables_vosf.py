import numpy as np
import tensorflow as tf
import gpflow

from gpflow import settings, transforms
from gpflow.features import InducingFeature, Kuu, Kuf
from gpflow.dispatch import dispatch
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow.params import Parameter, ParamList
from gpflow.kernels import Kernel, Combination, Sum, Product
from .kernels import SignatureKernel

from .kernels_pde import UntruncSignatureKernel 
from .utils import get_powers, compute_trunc


from iisignature_tensorflow import Sig
import iisignature

from tensorflow.python.framework import ops

class SignatureOrthogonalInducing(InducingFeature):
    """
    Base class for VOSF inducing variables for use with signature kernel in sparse variational GPs.
    """
    def __init__(self):
        super().__init__()


class UntruncInducingOrthogonalTensors(SignatureOrthogonalInducing):
    """
    # Inducing class for using the vosf inducing variables with the PDE signature kernel.
    # # Input
    # :input_dim:     the total size of an input sample to the kernel
    # :d:             the state-space dimension of the data streams
    # :M:             the number of inducing variables
    """

    def __init__(self, input_dim, d, M,compute_sig=False,num_lags=0,**kwargs):
       
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d = d
        self.num_lags = num_lags
    
        self.M = M
        self.compute_sig = compute_sig

        # compute the truncation level the "closest to M"
        self.sig_level = compute_trunc(M,(num_lags+1)*d)

    def __len__(self):
        return self.M

@dispatch(UntruncInducingOrthogonalTensors, UntruncSignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False, fast=False):
    ''' 
    In PDE-VOSF: 
        Kxx is computed with the PDE-signature kernel. 
        Kzz is the identity
        Kzx is a batch of signatures.
    '''

    with params_as_tensors_for(feat,kern):

        # Computing Kzz (the identity!)
        Kzz = Kuu(feat,kern) 

        # Computing Kzx 
        if fast: # this method is in development. It avoids the computation of signatures.
            Kzx = None 
        elif feat.compute_sig:
            Kzx = Kuf(feat,kern,X_new)
        else:
            Kzx = Kuf(feat,kern,X_new[:,feat.input_dim:]) # S(\theta x) using iisignature compatible with tensorflow even if we do not use its autodiff
        
        # Computing Kxx (with the PDE-signature kernel here)
        if full_f_cov:
            print('Not Implemented')
            Kxx = kern.Kdiag(X_new[:,:feat.input_dim]) # to change with full covariance matrix
            Kxx += jitter * tf.eye(tf.shape(X_new)[0], dtype=settings.dtypes.float_type)
        else:
            Kxx = kern.Kdiag(X_new[:,:feat.input_dim]) 
            Kxx += jitter

    return Kzz, Kzx, Kxx

@dispatch(UntruncInducingOrthogonalTensors, UntruncSignatureKernel, object)
def Kuf(feat, kern, X_new):
    ''' (Kuf)_{m,i}=S^m(x_i) 
    There are three options: 
    (1) the signatures are precomputed and concatenated with the input streams. 
    (2) we can use a trick to avoid computing any signatures (this constrains the form of the variational parameters)
    (3) we compute the signatures with iisignature (this option is now commented)
    ''' 

    with params_as_tensors_for(feat,kern):
        
        num_examples = tf.shape(X_new)[0]

        if feat.compute_sig:
            # X, _ = kern._slice(X_new, None)
            X = tf.reshape(X_new, (num_examples, -1, feat.d))
            X = kern._apply_scaling_and_lags_to_sequences(X)
            S_tf = Sig(X,feat.sig_level)
            S_tf = S_tf[:,:(feat.M-1)]  # (N,M)
        else:
            S_tf = X_new 
            indices = get_powers(feat.d,feat.sig_level)[:(feat.M-1),:] # this way we do not need to differentiate the signature wrt its input
            levels = tf.repeat(kern.lengthscales[None,:],repeats=tf.shape(indices)[0],axis=0)
            powers_levels = tf.pow(levels,indices)
            powers = tf.math.reduce_prod(powers_levels,axis=1)
            S_tf/=powers[None,:]
        ones = tf.ones([num_examples,1],dtype=settings.dtypes.float_type)
        full_S = tf.concat([ones,S_tf],axis=1)
        full_S *= tf.sqrt(kern.sigma)
        Kzx = tf.transpose(full_S)

    return Kzx

@dispatch(UntruncInducingOrthogonalTensors, UntruncSignatureKernel)
def Kuu(feat, kern):
    ''' Kuu = (<S^i(.),S^j(.)>)_i,j is diagonal since the signature features are orthogonal ''' 
    ''' The diagonal elements depend on the kernel hyperparameters kern.sigma and kern.lengthscale'''
    with params_as_tensors_for(feat,kern):
        Kzz = tf.eye(feat.M, dtype=settings.dtypes.float_type) 

    return Kzz

class TruncInducingOrthogonalTensors(SignatureOrthogonalInducing):
    """
    # Inducing class for using the vosf inducing variables with the truncated signature kernel.
    # # Input
    # :input_dim:     the total size of an input sample to the kernel
    # :d:             the state-space dimension of the data streams
    # :M:             the number of inducing variables
    """

    def __init__(self, input_dim, d, M,compute_sig=False,num_lags=0 **kwargs):
       
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d = d
        self.M = M
        self.num_lags = num_lags
        self.compute_sig = compute_sig

        # compute the truncation level the "closest to M"
        self.sig_level = compute_trunc(M,(num_lags+1)*d)

    def __len__(self):
        return self.M

@dispatch(TruncInducingOrthogonalTensors, SignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False, fast=False):
    ''' 
    In Trunc-VOSF: 
        Kxx is computed with the truncated signature kernel. 
        Kzz is the identity
        Kzx is a batch of signatures, which may be precomputed.
    '''

    with params_as_tensors_for(feat,kern):

        # Computing Kzz (the identity!)
        Kzz = Kuu(feat,kern)

        # Computing Kzx
        if fast: # in development
            Kzx = None
        elif feat.compute_sig:
            Kzx = Kuf(feat,kern,X_new)
        else:
            Kzx = Kuf(feat,kern,X_new[:,feat.input_dim:]) 

        # Computing Kxx (with the truncated signature kernel here)
        if full_f_cov:
            Kxx = kern.K(X_new[:,:feat.input_dim],return_levels = False)  
            Kxx += jitter * tf.eye(tf.shape(X)[0], dtype=settings.dtypes.float_type)
        else:
            if kern.normalization:
                Kxx_un = kern.K_norms(X_new[:,:feat.input_dim])
                Kzx /= tf.repeat(Kxx_un,repeats=np.array([feat.d**i for i in range(feat.sig_level+1)]),axis=0)
                Kzx *= tf.sqrt(kern.sigma)
            else:
                Kxx = kern.Kdiag(X_new[:,:feat.input_dim],return_levels = False) 
            Kxx += jitter
    return Kzz, Kzx, Kxx

@dispatch(TruncInducingOrthogonalTensors, SignatureKernel, object)
def Kuf(feat, kern, X_new):
    ''' (Kuf)_{m,i}=S^m(x_i) 
    There are three options: 
    (1) the signatures are precomputed and concatenated with the input streams. 
    (2) We can use a trick to avoid computing any signatures (this constrains the form of the variational parameters)
    (3) we compute the signatures with iisignature (this option is now commented)
    ''' 
    with params_as_tensors_for(feat,kern):
        num_examples = tf.shape(X_new)[0]

        if feat.compute_sig:
            # X, _ = kern._slice(X_new, None)
            X = tf.reshape(X_new, (num_examples, -1, feat.d))
            X = kern._apply_scaling_and_lags_to_sequences(X)
            S_tf = Sig(X,feat.sig_level)
            S_tf = S_tf[:,:(feat.M-1)]  # (N,M)
        else:
            S_tf = X_new
            indices = get_powers(feat.d,feat.sig_level)[:(feat.M-1),:]
            levels = tf.repeat(kern.lengthscales[None,:],repeats=tf.shape(indices)[0],axis=0)
            powers_levels = tf.pow(levels,indices)
            powers = tf.math.reduce_prod(powers_levels,axis=1)
            S_tf/=powers[None,:]

        ones = tf.ones([num_examples,1],dtype=settings.dtypes.float_type)

        full_S = tf.concat([ones,S_tf],axis=1)
        if not kern.normalization:
            full_S *= tf.sqrt(kern.sigma)
        Kzx = tf.transpose(full_S)
    return Kzx

@dispatch(TruncInducingOrthogonalTensors, SignatureKernel)
def Kuu(feat, kern):
    ''' Kuu = (<S^i(.),S^j(.)>)_i,j is diagonal since the signature features are orthogonal ''' 
    ''' The diagonal elements depend on the kernel hyperparameters kern.sigma and kern.lengthscale'''
    with params_as_tensors_for(feat,kern):
        Kzz = tf.eye(feat.M, dtype=settings.dtypes.float_type) 
    return Kzz
