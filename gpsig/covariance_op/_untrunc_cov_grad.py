"""The gradient of the UntruncCov op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

# need to find a better way to load the module
import os
import tensorflow as tf
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
path = find("untrunc_cov_op_gpu.so",'.')
cov_module_gpu = tf.load_op_library(path)

@ops.RegisterGradient("UntruncCov")
def _untrunc_cov_grad(op, grad1):
  """The gradients for `untrunc_cov`.

  Args:
    op: The `untrunc_cov` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `untrunc_cov` op.

  Returns:
    Gradients with respect to the input of `untrunc_cov`.
  """

  # get what we need to solve "reverse" pdes on cuda
  X = op.inputs[0]  
  E = op.inputs[1]
  sol = op.inputs[2]
  E_rev = tf.reverse(E,axis=[1],name='reverse1')
  E_rev = tf.reverse(E_rev,axis=[2],name='reverse2')

  # solve the "reverse" pdes on cuda
  K_rev_ = cov_module_gpu.untrunc_cov(X,E_rev, sol)
  K_rev = K_rev_[:,:-1,:-1]

  # get what we need to compute the gradients
  K_ = op.outputs[0]
  K = K_[:,:-1,:-1]
  paths_shape = array_ops.shape(X)
  
  # get shapes back
  A = paths_shape[0]
  M = paths_shape[1]
  D = paths_shape[2]

  # in principle we have only computed the lower triangular part of K_diag and K_diag_rev because they are symmetric as full_cov is false. 
  # however this has not been implemented yet on cuda
  # MM = tf.shape(K)[1]
  # diag_forward = tf.matrix_diag_part(K)
  # K += tf.transpose(K,perm=[0,2,1]) - diag_forward[:,:,None]*tf.eye(MM,dtype=tf.float64)[None,:,:]
  # diag_backward = tf.matrix_diag_part(K_rev)
  # K_rev += tf.transpose(K_rev,perm=[0,2,1]) - diag_backward[:,:,None]*tf.eye(MM,dtype=tf.float64)[None,:,:]

  inc_X = (X[:,1:,:]-X[:,:-1,:])                
  # Reorganize the K_rev matrix (maybe we could do this on cuda directly)
  K_rev_rev = tf.reverse(K_rev,axis=[1],name='reverse1')
  K_rev_rev = tf.reverse(K_rev_rev,axis=[2],name='reverse2')
  
  KK = (K[:,:-1,:-1] * K_rev_rev[:,1:,1:])                             
  K_grad = KK[:,:,:,None]*inc_X[:,None,:,:]                            
  K_grad = tf.reduce_sum(K_grad,axis=2)                                
  K_grad =  tf.reduce_sum(tf.reshape(K_grad,(A,M-1,1,D)),axis=2)       
  
  grad_points = -2.*tf.concat([K_grad,tf.zeros((A, 1, D),dtype=tf.float64)],axis=1) + 2.*tf.concat([tf.zeros((A, 1, D),dtype=tf.float64), K_grad], axis=1)

  return [grad1[:,-1,-1][:,None,None]*grad_points, None, None]

