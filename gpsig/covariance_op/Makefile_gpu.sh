#/bin/bash
CUDA_ROOT=/usr/local/cuda/
TF_ROOT=/anaconda/envs/sigGPDE_env/lib/python3.7/site-packages/tensorflow_core


ln -s ${TF_ROOT}/libtensorflow_framework.so.1 ${TF_ROOT}/libtensorflow_framework.so

${CUDA_ROOT}/bin/nvcc -std=c++11 -c -o untrunc_cov_op_gpu.cu.o untrunc_cov_op_gpu.cu -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 untrunc_cov_op_gpu.cc untrunc_cov_op_gpu.cu.o -o untrunc_cov_op_gpu.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L${TF_ROOT} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
