# cython: boundscheck=False
# cython: wraparound=False
from cython.parallel import prange
import numpy as np

cdef forward_step(double k_00, double k_01, double k_10, double increment):
	return k_10 + k_01 + k_00*(increment-1.)

cdef forward_step_explicit(double k_00, double k_01, double k_10, double increment):
	return (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)

cdef forward_step_implicit(double k_00, double k_01, double k_10, double increment):
	return k_01+k_10-k_00 + ((0.5*increment)/(1.-0.25*increment))*(k_01+k_10)

def sig_kern_diag(double[:,:,:] x, int n=0):
	cdef int A = x.shape[0]
	cdef int M = x.shape[1]
	cdef int D = x.shape[2]
	cdef int N = M
	cdef double increment, increment_rev
	cdef double factor = 2**(2*n)
	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)
	cdef double[:,:,:] K = np.zeros((A,MM+1,NN+1), dtype=np.float64)
	cdef double[:,:,:] K_rev = np.zeros((A,MM+1,NN+1), dtype=np.float64)
	
	for l in prange(A,nogil=True):
		for i in range(MM+1):
			K[l,i,0] = 1.
			K_rev[l,i,0] = 1.
			
		for i in range(MM):
			for j in range(i):
				ii = int(i/(2**n))
				jj = int(j/(2**n))
				
				increment = 0.
				increment_rev = 0.
				
				for k in range(D):
					increment = increment +  (x[l,ii+1,k]-x[l,ii,k])*(x[l,jj+1,k]-x[l,jj,k])/factor
					increment_rev = increment_rev + (x[l,(M-1)-(ii+1),k]-x[l,(M-1)-ii,k])*(x[l,(N-1)-(jj+1),k]-x[l,(N-1)-jj,k])/factor
		
				K[l,i+1,j+1] = (K[l,i,j+1] + K[l,i+1,j])*(1.+0.5*increment+(1./12)*increment**2) - K[l,i,j]*(1.-(1./12)*increment**2)
				K_rev[l,i+1,j+1] = (K_rev[l,i,j+1] + K_rev[l,i+1,j])*(1.+0.5*increment_rev+(1./12)*increment_rev**2) - K_rev[l,i,j]*(1.-(1./12)*increment_rev**2)
				#K_rev[l,i+1,j+1] = (K_rev[l,i,j+1] + K_rev[l,i+1,j]) + K_rev[l,i,j]*(increment_rev-1.)

			ii = int(i/(2**n))
			jj = int(i/(2**n))
			increment = 0.
			increment_rev = 0.
			for k in range(D):
				increment = increment +  (x[l,ii+1,k]-x[l,ii,k])*(x[l,jj+1,k]-x[l,jj,k])/factor
				increment_rev = increment_rev + (x[l,(M-1)-(ii+1),k]-x[l,(M-1)-ii,k])*(x[l,(N-1)-(jj+1),k]-x[l,(N-1)-jj,k])/factor
			K[l,i+1,i+1] = 2*(K[l,i+1,i])*(1.+0.5*increment+(1./12)*increment**2) - K[l,i,i]*(1.-(1./12)*increment**2)
			K_rev[l,i+1,i+1] = 2*(K_rev[l,i+1,i])*(1.+0.5*increment_rev+(1./12)*increment_rev**2) - K_rev[l,i,i]*(1.-(1./12)*increment_rev**2)

	return np.array(K), np.array(K_rev)

