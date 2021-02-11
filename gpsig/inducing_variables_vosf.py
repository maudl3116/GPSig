r"""
Two classes of orthogonal inducing variables (VOSF) for sparse GPs on sequential data. 
Both implement SignatureOrthogonalInducing:
    * One class is for GP models with the truncated signature kernel. 
    * Another is for GP models endowed with the PDE signature kernel.
The methods compute three types of covariance matrices that intervene in the ELBO: `Kxx, Kzx, Kzz`.
The respective shapes of these matrices are:
    * `Kxx` [N,N] computed with the signature kernel
    * `Kzz` [M,M] is the identity 
    * `Kzx` [N,M] is a batch of signatures 
"""

import numpy as np
import tensorflow as tf
import gpflow

from gpflow import settings, transforms
from gpflow.features import InducingFeature, Kuu, Kuf
from gpflow.dispatch import dispatch
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow.params import Parameter, ParamList
from gpflow.kernels import Kernel, Combination, Sum, Product
from tensorflow.python.framework import ops

from .kernels import SignatureKernel
from .kernels_pde import UntruncSignatureKernel 
from .utils import get_powers, compute_trunc
import iisignature
from iisignature_tensorflow import Sig

class SignatureOrthogonalInducing(InducingFeature):
    """
    Base class for VOSF inducing variables for use in GP models for sequential data.
    """
    def __init__(self):
        super().__init__()


class UntruncInducingOrthogonalTensors(SignatureOrthogonalInducing):
    """ 
    The inducing class for using the VOSF inducing variables with the PDE signature kernel.
    """

    def __init__(self, input_dim, d, M, compute_sig=False, compute_and_diff_sig=False, num_lags=0, **kwargs):
        """
        :param input_dim: the dimension of the input time seriess d*l
        :param d: the number of channels of the time series
        :param M: the number of inducing variables
        :param compute_sig: whether the signatures are computed on the fly and do not need AD (for simple parametrization of signature kernel)
        :param compute_and_diff_sig: whether the signatures are computed on the fly and differentiated with AD ( for complex parametrization of signature kernel)
        :param num_lags: the number of times all channels are lagged
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d = d
        self.num_lags = num_lags
    
        self.M = M
        self.compute_sig = compute_sig
        self.compute_and_diff_sig = compute_and_diff_sig

        # compute the truncation level the "closest to M"
        self.sig_level = compute_trunc(M,(num_lags+1)*d)

    def __len__(self):
        return self.M

@dispatch(UntruncInducingOrthogonalTensors, UntruncSignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False, fast=False):

    with params_as_tensors_for(feat,kern):

        # Computing Kzz (the identity!)
        Kzz = Kuu(feat,kern) 

        # Computing Kzx 
        if fast:                                                # TODO: trick in development to avoid the signature computation
            Kzx = None 
        elif feat.compute_sig or feat.compute_and_diff_sig:
            Kzx = Kuf(feat,kern,X_new)                          # uses iisignature to compute signatures on the fly
        else:
            Kzx = Kuf(feat,kern,X_new[:,feat.input_dim:])       # signatures are contained in the input
        
        # Computing Kxx (with the PDE-signature kernel)
        if full_f_cov:
            raise ValueError('Not implemented')
            Kxx = kern.Kdiag(X_new[:,:feat.input_dim])          # TODO: to change with full covariance matrix
            Kxx += jitter * tf.eye(tf.shape(X_new)[0], dtype=settings.dtypes.float_type)
        else:
            Kxx = kern.Kdiag(X_new[:,:feat.input_dim]) 
            Kxx += jitter

    return Kzz, Kzx, Kxx

@dispatch(UntruncInducingOrthogonalTensors, UntruncSignatureKernel, object)
def Kuf(feat, kern, X_new):
    """ 
    Computes Kzx such that (Kzx)_{m,i}=S^m(x_i) 
    
    There are three options: 
    
    * The signatures are precomputed, concatenated with the input streams and fed to the GP model 
        if `feat.compute_sig` and `feat.compute_and_diff_sig` are False
    
    * The signatures are computed with iisignature 
        if `feat.compute_sig` or `feat.compute_and_diff_sig` is True
    
    * We use a trick to avoid computing any signatures 
        if `feat.fast` is true
    """ 
    
    with params_as_tensors_for(feat,kern):
        
        num_examples = tf.shape(X_new)[0]
            
        if feat.compute_and_diff_sig:
            # COMPLEX PARAMETRIZATION SIG KERNEL
            # X, _ = kern._slice(X_new, None)
            X = tf.reshape(X_new, (num_examples, -1, feat.d))
            X = kern._apply_scaling_and_lags_to_sequences(X)
            S_tf = Sig(X,feat.sig_level)
            S_tf = S_tf[:,:(feat.M-1)]  # (N,M)
        else: 
            # SIMPLE ARD PARAMETRIZATION SIG KERNEL
            # (we can pull the parameters (`powers`) out of the signatures)
            indices = get_powers(feat.d,feat.sig_level)[:(feat.M-1),:] 
            levels = tf.repeat(kern.lengthscales[None,:],repeats=tf.shape(indices)[0],axis=0)
            powers_levels = tf.pow(levels,indices)
            powers = tf.math.reduce_prod(powers_levels,axis=1)
            if feat.compute_sig:
                # we compute the signatures on the fly with iisignature
                X = tf.reshape(X_new, (num_examples, -1, feat.d))
                S_tf = Sig(X,feat.sig_level)
                S_tf = S_tf[:,:(feat.M-1)]
            else:
                # the signatures have been precomputed
                S_tf = X_new 
            S_tf/=powers[None,:]

        # need to add the first signature feature which is always 1
        ones = tf.ones([num_examples,1],dtype=settings.dtypes.float_type)
        full_S = tf.concat([ones,S_tf],axis=1)
        full_S *= tf.sqrt(kern.sigma)
        Kzx = tf.transpose(full_S)

    return Kzx

@dispatch(UntruncInducingOrthogonalTensors, UntruncSignatureKernel)
def Kuu(feat, kern):
    with params_as_tensors_for(feat,kern):
        Kzz = tf.eye(feat.M, dtype=settings.dtypes.float_type) 
    return Kzz

class TruncInducingOrthogonalTensors(SignatureOrthogonalInducing):
    """ 
    The inducing class for using the VOSF inducing variables with the truncated signature kernel.
    """

    def __init__(self, input_dim, d, M, compute_sig=False, compute_and_diff_sig=False, num_lags=0, **kwargs):
        """
        :param input_dim: the dimension of the input time seriess d*l
        :param d: the number of channels of the time series
        :param M: the number of inducing variables
        :param compute_sig: whether the signatures are computed on the fly but do not need AD (for simple parametrization of signature kernel)
        :param compute_and_diff_sig: whether the signatures are computed on the fly and differentiated with AD ( for complex parametrization of signature kernel)
        :param num_lags: the number of times all channels are lagged
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d = d
        self.M = M
        self.num_lags = num_lags
        self.compute_sig = compute_sig
        self.compute_and_diff_sig = compute_and_diff_sig 

        # compute the truncation level the "closest to M"
        self.sig_level = compute_trunc(M,(num_lags+1)*d)

    def __len__(self):
        return self.M

@dispatch(TruncInducingOrthogonalTensors, SignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False, fast=False):

    with params_as_tensors_for(feat,kern):

        # Computing Kzz (the identity!)
        Kzz = Kuu(feat,kern)

        # Computing Kzx
        if fast:                                                        # TODO: trick in development to avoid the signature computation
            Kzx = None
        elif feat.compute_sig or feat.compute_and_diff_sig:
            Kzx = Kuf(feat,kern,X_new)                                  # uses iisignature to compute signatures on the fly
        else:
            Kzx = Kuf(feat,kern,X_new[:,feat.input_dim:])               # signatures are contained in the input

        # Computing Kxx (with the truncated signature kernel here)
        if full_f_cov:
            Kxx = kern.K(X_new[:,:feat.input_dim],return_levels = False)  
            Kxx += jitter * tf.eye(tf.shape(X_new)[0], dtype=settings.dtypes.float_type)
        else:
            if kern.normalization:
                # Kxx is a constant w.r.t  x, but here we need to get back the norms to corrext Kzx (the signatures). 
                Kxx, Kxx_un = kern.K_norms(X_new[:,:feat.input_dim])
                Kzx /= tf.repeat(tf.sqrt(Kxx_un[:(feat.sig_level+1)]+settings.jitter),repeats=np.array([feat.d**i for i in range(feat.sig_level+1)]),axis=0)[:feat.M,:]
            else:
                Kxx = kern.Kdiag(X_new[:,:feat.input_dim],return_levels = False) 
            Kxx += jitter
    return Kzz, Kzx, Kxx

@dispatch(TruncInducingOrthogonalTensors, SignatureKernel, object)
def Kuf(feat, kern, X_new):
    """ 
    Computes Kzx such that (Kzx)_{m,i}=S^m(x_i) 
    
    There are three options: 
    
    * The signatures are precomputed, concatenated with the input streams and fed to the GP model 
        if `feat.compute_sig` and `feat.compute_and_diff_sig` are False
    
    * The signatures are computed with iisignature 
        if `feat.compute_sig` or `feat.compute_and_diff_sig` is True
    
    * We use a trick to avoid computing any signatures 
        if `feat.fast` is true
    """ 
    with params_as_tensors_for(feat,kern):
        num_examples = tf.shape(X_new)[0]

        if feat.compute_and_diff_sig: 
            # COMPLEX PARAMETRIZATION SIG KERNEL
            # X, _ = kern._slice(X_new, None)
            X = tf.reshape(X_new, (num_examples, -1, feat.d))
            X = kern._apply_scaling_and_lags_to_sequences(X)
            S_tf = Sig(X,feat.sig_level)
            S_tf = S_tf[:,:(feat.M-1)]  # (N,M)
        else:
            # SIMPLE ARD PARAMETRIZATION SIG KERNEL
            # (we can pull the parameters (`powers`) out of the signatures)
            indices = get_powers(feat.d,feat.sig_level)[:(feat.M-1),:]
            levels = tf.repeat(kern.lengthscales[None,:],repeats=tf.shape(indices)[0],axis=0)
            powers_levels = tf.pow(levels,indices)
            powers = tf.math.reduce_prod(powers_levels,axis=1)
            if feat.compute_sig:
                # compute the signatures with iisignature 
                X = tf.reshape(X_new, (num_examples, -1, feat.d))
                S_tf = Sig(X,feat.sig_level)
                S_tf = S_tf[:,:(feat.M-1)]  # (N,M)
            else:
                # X_new is already the signatures 
                S_tf = X_new

            S_tf/=powers[None,:]

        # need to add the first signature feature which is always 1
        ones = tf.ones([num_examples,1],dtype=settings.dtypes.float_type)
        full_S = tf.concat([ones,S_tf],axis=1)
        Kzx = tf.transpose(full_S)

        Kzx *= tf.repeat(tf.sqrt(kern.variances[:(feat.sig_level+1),None]),repeats=np.array([feat.d**i for i in range(feat.sig_level+1)]),axis=0)[:feat.M,:] 
        Kzx *= tf.sqrt(kern.sigma)

        # note that if normalization, we need Kxx to return the true Kzx (hence use Kuu_Kuf_Kff)
    return Kzx

@dispatch(TruncInducingOrthogonalTensors, SignatureKernel)
def Kuu(feat, kern):
    with params_as_tensors_for(feat,kern):
        Kzz = tf.eye(feat.M, dtype=settings.dtypes.float_type) 
    return Kzz