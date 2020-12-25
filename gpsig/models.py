import numpy as np
import tensorflow as tf

from gpflow import settings, transforms, models, likelihoods, mean_functions
from gpflow.params import Parameter, DataHolder, Minibatch
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow.kullback_leiblers import gauss_kl
from gpflow.conditionals import base_conditional, _expand_independent_outputs
from gpflow.features import Kuu, Kuf

from .inducing_variables import InducingTensors, InducingSequences, Kuu_Kuf_Kff, Kuu, Kuf

# added these imports
from .inducing_variables_vosf import TruncInducingOrthogonalTensors, UntruncInducingOrthogonalTensors, Kuu_Kuf_Kff, Kuu, Kuf
from typing import Optional
from .utils import get_powers
from .kernels import SignatureLinear

class SVGP(models.SVGP):
    """
    Re-implementation of SVGP from GPflow with a few minor tweaks. Slightly more efficient with SignatureKernels, and when using the low-rank option with signature kernels, this code must be used.
    """
    def __init__(self, X, Y, kern, likelihood, feat, mean_function=None, num_latent=None, q_diag=False, whiten=True, minibatch_size=None, num_data=None, q_mu=None, q_sqrt=None, shuffle=True, fast_algo=False, **kwargs):

        if not isinstance(feat, InducingTensors) and not isinstance(feat, InducingSequences) and not isinstance(feat, InducingSequences) and not isinstance(feat, TruncInducingOrthogonalTensors) and not isinstance(feat, UntruncInducingOrthogonalTensors):
            raise ValueError('feat must be of type either InducingTensors, InducingSequences, TruncInducingOrthogonalTensors or UntruncInducingOrthogonalTensors')

        num_inducing = len(feat)

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, shuffle=shuffle, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, shuffle=shuffle, seed=0)

        models.GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = feat
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)
        self.fast_algo = fast_algo # added this

        # The option "fast_algo" is under development
        if self.fast_algo:
            # restrictions
            assert self.q_diag, "fast algorithm only works with diagonal variational covariance"
            assert num_inducing==np.sum([ self.feature.d**k for k in range(self.feature.sig_level+1) ]), "cannot use the fast algorithm if we have a number of inducing features not corresponding to a truncation level"
            
            # create auxiliary kernel to compute the approximate posterior mean and variance
            self.kern_auxiliary = SignatureLinear(X.shape[1], self.feature.d, num_levels=self.feature.sig_level, order=self.feature.sig_level, lengthscales=None,normalization=False, difference=True)
            # inherit the parameters of the GP model kernel
            self.kern_auxiliary.lengthscales = self.kern.lengthscales
            self.kern_auxiliary.sigma = self.kern.sigma

            # reparametrizing the variational means q_mu as sparse tensors
            q_mu = np.zeros((int(self.feature.sig_level*(self.feature.sig_level+1)/2), num_latent, self.feature.d))
            self.q_mu = Parameter(q_mu, dtype=settings.dtypes.float_type)  
            # need to separate the first component
            bias = np.zeros((1,num_latent))
            self.bias = Parameter(bias, dtype=settings.dtypes.float_type)  

            # reparametrizing the diagonal variational covariances q_sqrt as parse tensors
            q_sqrt = np.zeros((int(self.feature.sig_level*(self.feature.sig_level+1)/2), num_latent, self.feature.d)) #  q_srt was of of shape (M,num_latents), is now (N_M(N_M+1)/2,(M-1)*num_latent,d)
            self.q_sqrt = Parameter(q_sqrt, dtype=settings.dtypes.float_type,transform=transforms.positive) 
            first_var = np.zeros((1,num_latent))
            self.first_var = Parameter(first_var, dtype=settings.dtypes.float_type,transform=transforms.positive)  
            
    
    @params_as_tensors
    def _build_likelihood(self):

        X = self.X
        Y = self.Y

        num_samples = tf.shape(X)[0]

        if isinstance(self.feature, UntruncInducingOrthogonalTensors) or isinstance(self.feature, TruncInducingOrthogonalTensors):
            # we have 2 ways to compute f_mean and f_var:
                # method 1: with explicit signatures ( computing K_uf=S_M(X) )
                # method 2: using kernel tricks
            if self.fast_algo:
                
                f_mean, f_var = self._build_predict_fast(X, full_cov=False, full_output_cov=False, increments=False)
                
                # need to transform q_mu into a full (not sparse) tensor, and then flatten into a (num_inducing, num_latent) matrix  
                q_mu_flat = [tf.transpose(self.bias)] #(num_latent,1)
                k = 0
                for m in range(1, self.feature.sig_level+1):
                    Zm = self.q_mu[k]
                    k += 1
                    for i in range(1, m):
                        Zm = tf.reshape(Zm[..., None] * self.q_mu[k, :, None, :],[self.bias.shape[1], -1])
                        k += 1
                    q_mu_flat.append(Zm)
                q_mu_flat = tf.transpose(tf.concat(q_mu_flat, axis=1))
                # need to transform q_sqrt into a full (not sparse) tensor, and then flatten into a (num_inducing,num_latent) matrix
                
                q_sqrt_flat = [tf.transpose(self.first_var)] #(num_latent,1) 
                k = 0
                for m in range(1, self.feature.sig_level+1):
                    Zm = self.q_sqrt[k]
                    k += 1
                    for i in range(1, m):
                        Zm = tf.reshape(Zm[..., None] * self.q_sqrt[k, :, None, :],[self.q_sqrt.shape[1], -1])
                        k += 1
                    q_sqrt_flat.append(Zm)
                q_sqrt_flat = tf.transpose(tf.concat(q_sqrt_flat, axis=1))
                KL =  gauss_kl(q_mu_flat, q_sqrt_flat)  
            # added this. there is no need to use the whitening case, 
            else:
                f_mean, f_var = self._build_predict(X, full_cov=False, full_output_cov=False)
                KL =  gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0))
        else:
            if self.whiten:
                f_mean, f_var = self._build_predict(X, full_cov=False, full_output_cov=False)
                KL =  gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0))
            else:
                f_mean, f_var, Kzz = self._build_predict(X, full_cov=False, full_output_cov=False, return_Kzz=True)
                KL =  gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0), K=Kzz)
        
        # compute variational expectations
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        
        # scaling for batch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(num_samples, settings.float_type)
        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, X_new, full_cov=False, full_output_cov=False, return_Kzz=False):
        
        num_samples = tf.shape(X_new)[0]
        Kzz, Kzx, Kxx = Kuu_Kuf_Kff(self.feature, self.kern, X_new, jitter=settings.jitter, full_f_cov=full_cov)
        if isinstance(self.feature, InducingTensors) or isinstance(self.feature, InducingSequences):
            f_mean, f_var = base_conditional(Kzx, Kzz, Kxx, self.q_mu, full_cov=full_cov, q_sqrt=tf.matrix_band_part(self.q_sqrt, -1, 0), white=self.whiten)
        else:
            # added this. It is just to avoid cholesky decompositions we don't need with VOS
            f_mean, f_var = base_conditional_ortho(Kzx, Kzz, Kxx, self.q_mu, full_cov=full_cov, q_sqrt=tf.matrix_band_part(self.q_sqrt, -1, 0), white=self.whiten)
        f_mean += self.mean_function(X_new)
        f_var = _expand_independent_outputs(f_var, full_cov, full_output_cov)
        
        if return_Kzz:  
            return f_mean, f_var, Kzz
        else:
            return f_mean, f_var

    # This is in development
    @params_as_tensors
    def _build_predict_fast(self, X_new, full_cov=False, full_output_cov=False, return_Kzz=False, increments=True):
        
        num_samples = tf.shape(X_new)[0]
        Kzz, _, Kxx = Kuu_Kuf_Kff(self.feature, self.kern, X_new, jitter=settings.jitter, full_f_cov=full_cov, fast=True)

        # here we replace the base_conditional function
        f_mean = self.kern_auxiliary.K_tens_vs_seq(self.q_mu, X_new, increments=increments,apply_scalings=False) #(num_latent, num_examples)
        # the algorithm above does not take into account the first component of the mean, here we to correct for this.
        f_mean = tf.transpose(f_mean) - 1. + self.bias #(num_examples, num_latent)
        
        f_var =  self.kern_auxiliary.Kdiag_rescaled(self.q_sqrt, X_new)  # (num_examples, num_latent)
        f_var += -1. + 1.- self.first_var  
        f_var = Kxx[:,None] - f_var
        
        f_mean += self.mean_function(X_new)
        f_var = _expand_independent_outputs(f_var, full_cov, full_output_cov)
        
        if return_Kzz:  
            return f_mean, f_var, Kzz
        else:
            return f_mean, f_var

# added this
def base_conditional_ortho(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov=False,
    q_sqrt: Optional[tf.Tensor] = None,
    white=False,
):
    r"""
    Adapts the gpflow method base_conditional, to the case where the Kmm matrix is the identity. The aim is to
    speed up computations, by avoiding computing unnecessary cholesky decompositions
    """
    Lm = Kmm # no need for cholesky herem changed #tf.linalg.cholesky(Kmm)
    return base_conditional_with_lm_ortho(
        Kmn=Kmn, Lm=Lm, Knn=Knn, f=f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )
# added this
def base_conditional_with_lm_ortho(
    Kmn: tf.Tensor,
    Lm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov=False,
    q_sqrt: Optional[tf.Tensor] = None,
    white=False,
):
    r"""
    Has the same functionality as the `base_conditional` function, except that instead of
    `Kmm` this function accepts `Lm`, which is the Cholesky decomposition of `Kmm`.
    This allows `Lm` to be precomputed, which can improve performance.
    
    Adapts the gpflow method base_conditional_with_lm, to the case where the Kmm matrix is the identity. 
    The aim is to speed up computations, by avoiding computing unnecessary cholesky decompositions, and matrices
    multiplications.
    """
    # compute kernel stuff
    num_func = tf.shape(f)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f)[-2]

    # get the leading dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat(
        [
            tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K - 1, [1]),
        ],
        0,
    )  # [N]
    Kmn = tf.transpose(Kmn, perm)  # [..., M, N]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),
        (Lm, ["M", "M"]),
        (Knn, [..., "N", "N"] if full_cov else [..., "N"]),
        (f, ["M", "R"]),
    ]
    if q_sqrt is not None:
        shape_constraints.append(
            (q_sqrt, (["M", "R"] if q_sqrt.shape.ndims == 2 else ["R", "M", "M"]))
        )
    tf.debugging.assert_shapes(
        shape_constraints,
        message="base_conditional() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    leading_dims = tf.shape(Kmn)[:-2]

    # Compute the projection matrix A
    Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., M, M]
    A = Kmn # changed tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
        cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -3), cov_shape)  # [..., R, N, N]
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
        cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [..., R, N]

    # another backsubstitution in the unwhitened case
    #if not white:
     #   A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [..., M, R]
    f = tf.broadcast_to(f, f_shape)  # [..., M, R]
    fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

    if q_sqrt is not None:
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        elif q_sqrt_dims == 3:
            L = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
            L_shape = tf.shape(L)
            L = tf.broadcast_to(L, tf.concat([leading_dims, L_shape], 0))

            shape = tf.concat([leading_dims, [num_func, M, N]], axis=0)
            A_tiled = tf.broadcast_to(tf.expand_dims(A, -3), shape)
            LTA = tf.linalg.matmul(L, A_tiled, transpose_a=True)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))

        if full_cov:
            fvar = fvar + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [R, N]

    if not full_cov:
        fvar = tf.linalg.adjoint(fvar)  # [N, R]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),  # tensor included again for N dimension
        (f, [..., "M", "R"]),  # tensor included again for R dimension
        (fmean, [..., "N", "R"]),
        (fvar, [..., "R", "N", "N"] if full_cov else [..., "N", "R"]),
    ]
    tf.debugging.assert_shapes(shape_constraints, message="base_conditional() return values")

    return fmean, fvar
