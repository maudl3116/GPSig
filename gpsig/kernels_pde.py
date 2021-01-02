import tensorflow as tf
import numpy as np

from gpflow.params import Parameter
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow import transforms
from gpflow import settings
from gpflow.kernels import Kernel

from .sigKer_fast import sig_kern_diag as sig_kern_diag 
from . import lags

from tensorflow.python.framework import ops
# need to find a better way to load the module
import os
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
path = find("untrunc_cov_op_gpu.so",'.')
cov_module_gpu = tf.load_op_library(path)
from covariance_op import _untrunc_cov_grad

from numba import cuda

class UntruncSignatureKernel(Kernel):
    """
    """
    def __init__(self, input_dim, num_features, lengthscales=1,
                 order=0, num_lags=None, implementation = 'gpu_op',name=None):
        """
        # Inputs:
        ## Args:
        :input_dim:         the total size of an input sample to the kernel
        :num_features:      the state-space dimension of the input sequebces,
        
        ## Kwargs:
        ### Kernel options     
		:lengthscales:      lengthscales for scaling the coordinates of the input sequences,
                            if lengthscales is None, no scaling is applied to the paths
                            if ARD is True, there is one lengthscale for each path dimension, i.e. lengthscales is of size (num_features)
        :order              corresponds to the level of discretization to solve the PDE to approximate the signature kernel
        :num_lags:          Nonnegative integer or None, the number of lags added to each sequence. Usually between 0-5.
        :implementation:    'cython' if we want to compute on the CPU. 'gpu_op' if we want to use a dedicated cuda tensorflow operator. 
                            The 'cython' option computes the kernel (cython code) and another kernel to be able to compute the gradients.
        ***** TO DO *****:
            allow num_lags to be different from None
        
        """
        
        super().__init__(input_dim, name=name)
        self.num_features = num_features
        self.len_examples = self._validate_number_of_features(input_dim, num_features)

        assert num_lags is None, "VOSF does not handle lags yet"
        assert implementation in ['cython', 'gpu_op'], "implementation should be 'cython' or 'gpu_op'"
        self.implementation = implementation
        self.order = order
        self.sigma = Parameter(1., transform=transforms.positive, dtype=settings.float_type)
        
        if num_lags is None:
            self.num_lags = 0	
        else:
            if not isinstance(num_lags, int) or num_lags < 0:
                raise ValueError('The variable num_lags most be a nonnegative integer or None.')
            else:
                self.num_lags = int(num_lags)
                if num_lags > 0:
                    self.lags = Parameter(0.1 * np.asarray(range(1, num_lags+1)), transform=transforms.Logistic(), dtype=settings.float_type)
                    gamma = 1. / np.asarray(range(1, self.num_lags+2))
                    gamma /= np.sum(gamma)                   
                    self.gamma = Parameter(gamma, transform=transforms.positive, dtype=settings.float_type)

        if lengthscales is not None:
            lengthscales = self._validate_signature_param("lengthscales", lengthscales, self.num_features)
            self.lengthscales = Parameter(lengthscales, transform=transforms.positive, dtype=settings.float_type)
        else:
            self.lengthscales = None

	######################
	## Input validators ##
	######################
	
    def _validate_number_of_features(self, input_dim, num_features):
        """
        Validates the format of the input samples.
        """
        if input_dim % num_features == 0:
            len_examples = int(input_dim / num_features)
        else:
            raise ValueError("The arguments num_features and input_dim are not consistent.")
        return len_examples

    def _validate_signature_param(self, name, value, length):
        """
        Validates signature params
        """
        value = value * np.ones(length, dtype=settings.float_type)
        correct_shape = () if length==1 else (length,)
        if np.asarray(value).squeeze().shape != correct_shape:
            raise ValueError("shape of parameter {} is not what is expected ({})".format(name, length))
        return value

    ########################################
    ## Autoflow functions for interfacing ##
    ########################################


    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def compute_K(self, X, Y):
        return self.K(X, Y)

    @autoflow((settings.float_type, [None, None]))
    def compute_K_symm(self, X):
        return self.K(X)

    @params_as_tensors
    def _apply_scaling_and_lags_to_sequences(self, X):
        """
        Applies scaling and lags to sequences.
        """
        
        num_examples, len_examples, _ = tf.unstack(tf.shape(X))
        
        num_features = self.num_features * (self.num_lags + 1)
        
        if self.num_lags > 0:
            X = lags.add_lags_to_sequences(X, self.lags)

        X = tf.reshape(X, (num_examples, len_examples, self.num_lags+1, self.num_features))
        
        if self.lengthscales is not None:
            X /= self.lengthscales[None, None, None, :]

        if self.num_lags > 0:
            X *= self.gamma[None, None, :, None]
        
        X = tf.reshape(X, (num_examples, len_examples, num_features))
        return X



    @params_as_tensors
    def Kdiag(self, X, presliced=False,name=None):
        """
        Computes the diagonal of a square signature kernel matrix. To be re-implemented, with custom autograd
        """

        num_examples = tf.shape(X)[0]
            
        if not presliced:
            X, _ = self._slice(X, None)

        X = tf.reshape(X, (num_examples, -1, self.num_features))
        X = self._apply_scaling_and_lags_to_sequences(X)       

        if self.implementation == 'cython':
            K_diag = Kdiag_python(X,self.order)
        elif self.implementation == 'gpu_op':
            incr = X[:,1:,:]-X[:,:-1,:]
            E = tf.matmul(incr,incr,transpose_b=True)
            sol = tf.ones([num_examples, tf.shape(X)[1]+1,tf.shape(X)[1]+1],dtype=settings.float_type)
            K_diag_ = cov_module_gpu.untrunc_cov(X,E, sol)
            K_diag = K_diag_[:,:-1,:-1]
        return self.sigma*K_diag[:,-1,-1]

''' Functions for the Cython covariance operator ''' 

def Kdiag_python(X,order=0,name=None):
    with ops.name_scope(name, "Kdiag_python", [X]) as name:
        K_diag, grad = py_func(sig_kern_diag,
                                 [X,order],
                                 [settings.float_type, settings.float_type],
                                 name = name,
                                 grad = _KdiagGrad
        )
        return K_diag


def py_func(func,inp,Tout, stateful=True, name=None, grad=None):

    rnd_name = 'PyFuncGrad' + str(np.random.randint(0,1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()

    with g.gradient_override_map({"PyFunc":rnd_name}):
        return tf.py_func(func,inp,Tout,stateful=stateful, name=name)

def _KdiagGrad(op, grad_next_op_Kdiag,grad_next_op_Kdiag_grad):
    
    ''' inputs:
        - op: is pyfunc
        - grad_next_op_Kdiag: <-> grad_{K_diag} K_diag[:,-1,-1]
        - we do not use grad_next_op_Kdiag_grad
    '''

    n = op.inputs[1]
    c = tf.math.pow(2,n)
    cfloat = tf.dtypes.cast(c,settings.float_type)

    X = op.inputs[0]
    K_diag = op.outputs[0]
    K_diag_rev = op.outputs[1]
    
    # get shapes back
    A = tf.shape(X)[0]
    M = tf.shape(X)[1]
    D = tf.shape(X)[2]
  
    # we have only computed the lower triangular part of K_diag and K_diag_rev because they are symmetric as full_cov is false
    MM = tf.shape(K_diag)[1]
    diag_forward = tf.matrix_diag_part(K_diag)
    K_diag += tf.transpose(K_diag,perm=[0,2,1]) - diag_forward[:,:,None]*tf.eye(MM,dtype=settings.float_type)[None,:,:]
    diag_backward = tf.matrix_diag_part(K_diag_rev)
    K_diag_rev += tf.transpose(K_diag_rev,perm=[0,2,1]) - diag_backward[:,:,None]*tf.eye(MM,dtype=settings.float_type)[None,:,:]

    # to compute the real gradient, as K_diag_rev is only one step of that computation
    inc_X = (X[:,1:,:]-X[:,:-1,:])/cfloat  #(A,M-1,D)  increments defined by the data                
    inc_X = tf.repeat(inc_X, repeats=2**n, axis=1) #(A,(2**n)*(M-1),D)  increments on the finer grid
    
    # Reorganize the K_rev matrix
    K_rev_rev = tf.reverse(K_diag_rev,axis=[1],name='reverse1')
    K_rev_rev = tf.reverse(K_rev_rev,axis=[2],name='reverse2')

    KK = (K_diag[:,:-1,:-1] * K_rev_rev[:,1:,1:])                   # (A,(2**n)*(M-1),(2**n)*(N-1))

    K_grad = KK[:,:,:,None]*inc_X[:,None,:,:]                       # (A,(2**n)*(M-1),(2**n)*(N-1),D)
    
    n = tf.dtypes.cast(n, settings.float_type)

    K_grad = (1./cfloat)*tf.reduce_sum(K_grad,axis=2)               # (A,(2**n)*(M-1),D)

    K_grad =  tf.reduce_sum(tf.reshape(K_grad,(A,M-1,c,D)),axis=2)  # (A,M-1,D)

    # The gradient { grad_{X_i}[K(X_i,X_i)] : shape M,D } -> shape A,M,D 
    # we need to multiply by 2, because we have only computed the left gradient of the bilinear function
    grad_points = -2.*tf.concat([K_grad,tf.zeros((A, 1, D),dtype=settings.float_type)],axis=1) + 2.*tf.concat([tf.zeros((A, 1, D),dtype=settings.float_type), K_grad], axis=1)
    
    return grad_next_op_Kdiag[:,-1,-1][:,None,None]*grad_points, None

