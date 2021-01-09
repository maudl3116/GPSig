import numpy as np
import tensorflow as tf

from gpflow import settings

def signature_kern_rescaled_higher_order(M, num_levels):
    """
    Computes all the S_n(X)^T (I-Lambda_r)S_n(X) for "num_tensors" different diagonal matrices Lambda_r which can be parametrized as sparse tensors
    # Input
    :M:                 (num_examples, len_examples, len_tensors, 2*num_tensors, len_examples) tensors (each sparse tensor (diagonal matrix) is accompanied by an identity tensor)
    :num_levels:        number of signature levels to compute
    :order:             order of approximation to use in signature kernel
    # Output
    :K:                 (num_examples,num_tensors) tensor
    """

    num_tensors, num_examples = tf.shape(M)[3], tf.shape(M)[0]
    K = [tf.ones((num_examples, num_tensors), dtype=settings.float_type)]
    
    M = M[:, 1:, ..., 1:] + M[:, :-1, ..., :-1] - M[:, :-1, ..., 1:] - M[:, 1:, ..., :-1]
 
    r = 0
    
    for i in range(1, num_levels+1):
        R = np.asarray([[M[:,:,r,:,:]]])
        r += 1
        for j in range(1, i):
            d = min(j+1, num_levels)
            R_next = np.empty((d,d), dtype=tf.Tensor)
            R_next[0,0] = M[:,:,r,:,:] * tf.cumsum(tf.cumsum(tf.add_n(R.flatten().tolist()), exclusive=True, axis=1), exclusive=True, axis=-1)
            for l in range(1, d):
                R_next[0, l] = 1 / tf.cast(l+1, settings.float_type) * M[:,:,r,:,:] * tf.cumsum(tf.add_n(R[:,l-1].tolist()), exclusive=True, axis=1)
                R_next[l, 0] = 1 / tf.cast(l+1, settings.float_type) * M[:,:,r,:,:] * tf.cumsum(tf.add_n(R[l-1, :].tolist()), exclusive=True, axis=-1)
                for k in range(1, d):
                    R_next[l, k] = 1 / (tf.cast(l+1, settings.float_type) * tf.cast(k+1, settings.float_type)) * M[:,:,r,:,:] * R[l-1, k-1]
   
            R = R_next

            r += 1
        
        K.append(tf.reduce_sum(tf.add_n(R.flatten().tolist()), axis=(1,-1)))
    K = [-e[:,:num_tensors//2]+e[:,num_tensors//2:] for e in K] # in order to get (I-Lambda_r)...
    return tf.stack(K, axis=0)


def tensor_inner_product(M, num_levels):
    """
    Computing the vector of inner products of inducing tensors  <z_i,v_i>. 
    # Input
    :M:                 (num_levels*(num_levels+1)/2, num_tensors) 
    :num_levels:        degree of truncation for the signatures
    # Output
    :K:                 (num_levels, num_tensors) vector of tensor inner products per level 
    """

    num_tensors = tf.shape(M)[1]
    
    K = [tf.ones(num_tensors, dtype=settings.float_type)]
    
    k = 0
    for i in range(1, num_levels+1):
        R = M[k]
        k += 1
        for j in range(1, i):
            R = M[k] * R
            k += 1
        K.append(R)

    return tf.stack(K, axis=0)

def tensor_logs(M, num_levels, d):
    """
    Computing the vector of inner products of inducing tensors  <z_i,v_i>. 
    # Input
    :M:                 (num_levels*(num_levels+1)/2, num_tensors) 
    :num_levels:        degree of truncation for the signatures
    :d                  feature_dim
    # Output
    :K:                 (num_levels, num_tensors) vector of tensor inner products per level 
    """

    num_tensors = tf.shape(M)[1]
    
    K = [tf.zeros(num_tensors, dtype=settings.float_type)]
    
    k = 0
    for i in range(1, num_levels+1):
        R = M[k]
        k += 1
        for j in range(1, i): # can be done more efficiently
            R = M[k] + R
            k += 1
        K.append(tf.cast(d**(i-1), settings.float_type)*R)

    return tf.stack(K, axis=0)