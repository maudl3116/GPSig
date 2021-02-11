import tensorflow as tf
import numpy as np
import sys, os
from gpflow.params import Parameter
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow import transforms
from gpflow import settings
from gpflow.kernels import Kernel
from . import signature_algs_vosf
from . import signature_algs
from .sigKer_fast import sig_kern_diag as sig_kern_diag 
from . import lags
from tensorflow.python.framework import ops

# TODO: need to find a better way to load the module

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
try:
    path = find("untrunc_cov_op_gpu.so",'..')
    cov_module_gpu = tf.load_op_library(path)
    sys.path.append('../gpsig')
    from covariance_op import _untrunc_cov_grad
    print('Successfully loaded the Cuda PDE signature kernel operator')
except:
    pass



class UntruncSignatureKernel(Kernel):
    """
    Base class for the PDE signature kernel
    """
    def __init__(self, input_dim, num_features, 
                 lengthscales=1, order=0, num_lags=None, implementation = 'gpu_op', num_levels=None, name=None):
        """
        :param input_dim: the total size of an input sample to the kernel
        :param num_features: the number of channels the input sequebces,
    
		:param lengthscales: lengthscales for scaling the channels of the input sequences,
        :param order: corresponds to the level of discretization to solve the PDE to approximate the signature kernel
        :param num_lags: nonnegative integer or None, the number of lags added to each sequence. Usually between 0-5.
        :param implementation: 'cython' if we want to compute on the CPU. 'gpu_op' if we want to use a dedicated cuda tensorflow operator. 
        :param num_levels: the truncation level of the signatures (related to the number of inducing variables) for the trick that avoids computing any signature
        """
        
        super().__init__(input_dim, name=name)
        self.num_features = num_features
        self.len_examples = self._validate_number_of_features(input_dim, num_features)
        
        assert implementation in ['cython', 'gpu_op'], "implementation should be 'cython' or 'gpu_op'"
        self.implementation = implementation
        self.order = order
        if implementation=='gpu_op':
            assert ((2**self.order)*(self.len_examples-1)+ 1) < 1024, "discretization level of the PDE solver too large, to use the GPU operator"
        self.sigma = Parameter(1., transform=transforms.positive, dtype=settings.float_type)
        self.num_levels = num_levels

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
        return self.Kdiag(X)

    @autoflow((settings.float_type,), (settings.float_type, [None, None]))
    def compute_inner_product_tens_vs_seq(self, Z, X): 
        return self.inner_product_tens_vs_seq(Z, X)

    @autoflow((settings.float_type,), (settings.float_type, [None, None]))
    def compute_mahalanobis_terms_approx_posterior(self, Z, X): 
        return self.Mahalanobis_term_approx_posterior(Z, X)

    @autoflow((settings.float_type, ))
    def compute_norms_tens(self, Z): 
        return self.norms_tens(Z)

    @autoflow((settings.float_type, ))
    def compute_logs_tens(self, Z): 
        return self.logs_tens(Z)

    @autoflow((settings.float_type, [None, None,None]))
    def compute_K_base(self, X):
        return self._base_kern(X)


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
            E = tf.matmul(X,X,transpose_b=True) 
            # E = self._base_kern(X)
            E = E[:, 1:, ..., 1:] + E[:, :-1, ..., :-1] - E[:, :-1, ..., 1:] - E[:, 1:, ..., :-1]
            if self.order>0:
                E = tf.repeat(tf.repeat(E, repeats=2**self.order, axis=1)/tf.cast(2**self.order, settings.float_type), repeats=2**self.order, axis=2)/tf.cast(2**self.order, settings.float_type)
            sol = tf.ones([num_examples, (2**self.order)*(tf.shape(X)[1]-1)+2,(2**self.order)*(tf.shape(X)[1]-1)+2],dtype=settings.float_type)
            K_diag_ = cov_module_gpu.untrunc_cov(X,E, sol,self.order)
            K_diag = K_diag_[:,:-1,:-1]         

        return self.sigma*K_diag[:,-1,-1]

	####################################################################
	## Methods for the trick to avoid computing any signature in VOSF ##
	####################################################################

    def _Mahalanobis_term_approx_posterior(self, Z, X):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) 
        :X:             (num_examples, len_examples, num_features)
        # Output
        :K:             (num_levels+1, num_examples) 
        """
        
        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]
        num_examples, len_examples = tf.shape(X)[-3], tf.shape(X)[-2]
    
        Z = tf.concat([Z, tf.ones_like(Z)],axis=1)  
        
        ## linear embedding
        # ZX = Z[:,:,None,None,:]*X[None,None,:,:,:]
        # ZX = tf.transpose(ZX,perm=[2,0,1,3,4]) #(num_examples, len_tensors, 2*num_tensors, len_examples,num_features)
        # ZX = tf.reshape(ZX,[num_examples, len_tensors*2*num_tensors*len_examples, num_features])
        # M = tf.matmul(X,ZX,transpose_b=True) 
        # M = tf.reshape(M, (num_examples, len_examples, len_tensors, 2*num_tensors,len_examples))

        ## rbf embedding
        X = tf.transpose(X,perm=[0,2,1])
        M = self._base_kern(tf.reshape(X,[-1,len_examples,1]))  # (num_examples*d, len_examples, len_examples)
        M = tf.reshape(M,[num_examples, num_features,len_examples,len_examples])  # M[i,d,p,q] = <[x^i_p]_d,[x^j_p]_d>
        M = tf.transpose(M,perm=[0,3,2,1])  # M[i,p,q,d] = <[x^i_p]_d,[x^j_p]_d>
        M = M[:,:,None,None,:,:]*Z[None,None,:,:,None,:] 
        M = tf.reduce_sum(M,axis=-1)

        K_lvls_diag = signature_algs_vosf.signature_kern_rescaled_higher_order(M, self.num_levels)
        
        return K_lvls_diag

    def _Mahalanobis_tens(self, Z, beta):
        """
        # Input
        :beta:          (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors
        # Output
        :K:             (num_levels+1, num_tensors) 
        """
        
        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]

        M = self._base_kern(tf.reshape(beta,[-1,1,1]))  # (len_tensors*num_tensors*num_features, 1, 1)
        M = tf.reshape(M,[len_tensors, num_tensors, num_features])  
        M = M*Z
        M = tf.reduce_sum(M,axis=-1)  #[len_tensors, num_tensors]

        K_lvls_diag = signature_algs_vosf.tensor_inner_product(M, self.num_levels)
        
        return K_lvls_diag

    def _norms_tens(self, Z, embedding=True):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors
        # Output
        :K:             (num_levels+1, num_examples) tensor of (unnormalized) diagonals of signature kernel
        """
        
        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]
        
        ## lin embedding
        # M = tf.reduce_sum(tf.square(Z),axis=2)
        
        ## rbf embedding
        if embedding:
            M = tf.reshape( self._base_kern(tf.reshape(Z,[-1,1,num_features])), [len_tensors,num_tensors])
        else:
            M = tf.reduce_sum(tf.square(Z),axis=2)   
        
        K_lvls_diag = signature_algs_vosf.tensor_inner_product(M, self.num_levels)
        
        return K_lvls_diag

    def _logs_tens(self, Z):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors
        # Output
        :K:             (num_levels+1, num_examples) tensor of (unnormalized) diagonals of signature kernel
        """
        
        M = tf.reduce_sum(tf.log(Z),axis=2) # (num_levels*(num_levels+1)/2, num_tensors)
    
        K_lvls_diag = signature_algs_vosf.tensor_logs(M, self.num_levels, tf.shape(Z)[2])
        
        return K_lvls_diag

    def _K_tens_vs_seq_vosf(self, Z, X):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors, if not increments 
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        :X:             (num_examples, len_examples, num_features) tensor of sequences 
        Output
        :K_lvls:        (num_levels+1,) list of inducing tensors vs input sequences covariance matrices on each T.A. level
        """
        
        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]
        num_examples, len_examples = tf.shape(X)[-3], tf.shape(X)[-2]

        X = tf.reshape(X, [num_examples * len_examples, num_features])
 
        Z = tf.reshape(Z, [num_tensors * len_tensors, num_features])

        ## no embedding
        # M = tf.matmul(Z,X,transpose_b=True)
        # M = tf.reshape(M, (len_tensors, num_tensors, num_examples, len_examples))
        
        ## rbf embedding
        M = tf.reshape(self._base_kern(Z, X), (len_tensors, num_tensors, num_examples, len_examples))
        
        K_lvls = signature_algs.signature_kern_tens_vs_seq_higher_order(M, self.num_levels, order=self.num_levels, difference=True)
        
        return K_lvls

    @params_as_tensors
    def Mahalanobis_term_approx_posterior(self, Z, X, presliced=False):
        """
        Computes diag( S(X)^T(I-\Lambda_r)S(X)^T )  for different matrices \Lambda_r which are represented as rank-1 tensors.
        -> to rename
        """

        num_examples = tf.shape(X)[0]
        
        if not presliced:
            X, _ = self._slice(X, None)

        X = tf.reshape(X, (num_examples, -1, self.num_features))

        X = self._apply_scaling_and_lags_to_sequences(X)

        K_lvls_diag = self._Mahalanobis_term_approx_posterior(Z[1:],X)

        K_lvls_diag *= self.sigma          
        
        return tf.reduce_sum(K_lvls_diag, axis=0) + 1. - Z[0,:,0][None,:]

    @params_as_tensors
    def Mahalanobis_tens(self, Z, beta):
        """
        Computes diag( S(X)^T(I-\Lambda_r)S(X)^T )  for different matrices \Lambda_r which are represented as rank-1 tensors.
        -> to rename
        """

        K_lvls_diag = self._Mahalanobis_tens(Z[1:],beta[1:])
          
        return tf.reduce_sum(K_lvls_diag, axis=0) - 1. + (Z[0,:,0]*beta[0,:,0]**2)[None,:]



    @params_as_tensors 
    def norms_tens(self, Z, embedding=True):
        """
        Computes the vector of k_phi(z^i,z^i) for z^i in Z
        """
        constant_term = Z[0,:,0]

        K_lvls = self._norms_tens(Z[1:],embedding=embedding) 
        
        return tf.reduce_sum(K_lvls, axis=0) -1. + constant_term**2

    @params_as_tensors 
    def logs_tens(self, Z):
        """
        Computes the sum of the components of the tensors z^i in Z
        """
        constant_term = Z[0,:,0]

        K_lvls = self._logs_tens(Z[1:]) 
        
        return tf.reduce_sum(K_lvls, axis=0) + tf.log(constant_term)

    @params_as_tensors
    def inner_product_tens_vs_seq(self, Z, X, presliced=False):
        """
        Computes < S(X), m_r > for different tensors m_r 
        """

        if not presliced:
            X, _ = self._slice(X, None)
        
        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, (num_examples, -1, self.num_features))
        len_examples = tf.shape(X)[1]
        
        num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0] - 1

        X = self._apply_scaling_and_lags_to_sequences(X)

        Kzx_lvls = self._K_tens_vs_seq_vosf(Z[1:], X)  
        
        Kzx_lvls *= tf.sqrt(self.sigma) 

        return tf.reduce_sum(Kzx_lvls, axis=0) + tf.sqrt(self.sigma) *(Z[0,:,0][:,None]-1.)

    ##### Helper functions for base kernels

    def _square_dist(self, X, X2=None):
        batch = tf.shape(X)[:-2]
        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, tf.concat((batch, [-1, 1]), axis=0))  + tf.reshape(Xs, tf.concat((batch, [1, -1]), axis=0))
            return dist

        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, tf.concat((batch, [-1, 1]), axis=0)) + tf.reshape(X2s, tf.concat((batch, [1, -1]), axis=0))
        return dist

class SignatureRBF(UntruncSignatureKernel):
    """
    The signature kernel, which uses an (infinite number of) monomials of vectors - i.e. Gauss/RBF/SquaredExponential kernel - as state-space embedding
    """
    def __init__(self, input_dim, num_features, order, num_levels, **kwargs):
        UntruncSignatureKernel.__init__(self, input_dim, num_features, order, **kwargs)
        self._base_kern = self._rbf
        self.num_levels = num_levels

    # __init__.__doc__ = UntruncSignatureKernel.__init__.__doc__

    def _rbf(self, X, X2=None):
        K = tf.exp(-self._square_dist(X, X2) / 2)
        return K 


class SignatureLinear(UntruncSignatureKernel):
    """
    The signature kernel, which uses the identity as state-space embedding 
    """

    def __init__(self, input_dim, num_features, order, num_levels, **kwargs):
        UntruncSignatureKernel.__init__(self, input_dim, num_features, order, **kwargs)
        self._base_kern = self._lin
        self.num_levels = num_levels
    
    # __init__.__doc__ = UntruncSignatureKernel.__init__.__doc__

    def _lin(self, X, X2=None):
        if X2 is None:
            K = tf.matmul(X, X, transpose_b = True)
            return  K
        else:
            return tf.matmul(X, X2, transpose_b = True)


##############################################################################
## Methods to enable computation and differentiation of the cython operator ##
##############################################################################

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

