#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void UntruncCovKernel(
  const double* incr,
  const int nb_incr,
  const int paths_length,
  const int paths_length_,
  const int nb_diagonals,
  double* pdes_sol)

{
  
  unsigned int p =  blockIdx.x;
  unsigned int idx = threadIdx.x;

  for (int diag=0; diag<nb_diagonals; diag++){

      unsigned int J = max(0, min(diag - idx, paths_length - 1));

      unsigned int i = idx + 1;
      unsigned int j = J + 1;

      if( idx+J==diag && (idx<paths_length && J<paths_length)){
        
        float increment = incr[(nb_incr*nb_incr)*p + (i-1)*nb_incr + (j-1)];

        pdes_sol[(paths_length_*paths_length_)*p + i*paths_length_ + j] = ( pdes_sol[(paths_length_*paths_length_)*p + (i-1)*paths_length_ + j] + pdes_sol[(paths_length_*paths_length_)*p + i*paths_length_ + j-1] )*(1.+0.5*increment+(1./12)*increment*increment) - pdes_sol[(paths_length_*paths_length_)*p + (i-1)*paths_length_ + j-1]*(1.-(1./12)*increment*increment);
 
        }
    
    __syncthreads();
  }
      
}
	


void UntruncCovKernelLauncher(
  const double* incr,
  const int batch_samples,
  const int nb_incr,
  const int paths_length,
  const int paths_length_,
  const int nb_diagonals,
  double* pdes_sol)
{
  UntruncCovKernel<<<batch_samples, paths_length>>>(incr,nb_incr,paths_length,paths_length_,nb_diagonals,pdes_sol);
}
