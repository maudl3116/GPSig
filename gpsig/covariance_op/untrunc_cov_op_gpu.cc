#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/* 
  Register UntrucCov operation
  to do: enable possibility to have n!=0
*/

REGISTER_OP("UntruncCov")
  .Input("paths: double")
  .Input("incr: double")
  .Input("sol: double")
  .Input("order: int32")
  .Output("pdes_sol: double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    // input 0 is just because we need to access it for the gradients 
    shape_inference::ShapeHandle incr_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2),3,&incr_shape));
    return Status::OK();

  });


REGISTER_OP("UntruncCovRev")
  .Input("paths: double")
  .Input("incr: double")
  .Input("sol: double")
  .Input("order: int32")
  .Output("pdes_sol: double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    // input 0 is just because we need to access it for the gradients 
    shape_inference::ShapeHandle incr_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2),3,&incr_shape));
    return Status::OK();

  });


void UntruncCovKernelLauncher(
  const double* incr,
  const int batch_samples,
  const int nb_incr,
  const int paths_length,
  const int paths_length_,
  const int nb_diagonals,
  double* pdes_sol);

void UntruncCovRevKernelLauncher(
  const double* incr,
  const int batch_samples,
  const int nb_incr,
  const int paths_length,
  const int paths_length_,
  const int nb_diagonals,
  double* pdes_sol);

class UntruncCovOp: public OpKernel {
public:
  explicit UntruncCovOp(OpKernelConstruction* context) : OpKernel(context){
  }

  void Compute(OpKernelContext* context) override {
    
    // get the input tensors
    const Tensor& M = context->input(1);
    Tensor init = context->input(2);
   
    // get shapes
    const TensorShape& M_shape = M.shape();    
  
    const int batch_samples = M_shape.dim_size(0);
    const int nb_incr = M_shape.dim_size(1); 
    const int paths_length = M_shape.dim_size(1)+1; 
    const int nb_diagonals = 2*paths_length -1; 
    const int paths_length_ = M_shape.dim_size(1)+2;


    // check that the incr are 3 dimensional
    DCHECK_EQ(M_shape.dims(),3);
    
    context->set_output(0, context->input(2));
    
    auto f_M = M.flat<double>();
    auto f_sol = init.flat<double>();

    
    UntruncCovKernelLauncher(
      f_M.data(),
      batch_samples,
      nb_incr,
      paths_length,
      paths_length_,
      nb_diagonals,
      f_sol.data()
    );

    }
  };


class UntruncCovRevOp: public OpKernel {
public:
  explicit UntruncCovRevOp(OpKernelConstruction* context) : OpKernel(context){
  }

  void Compute(OpKernelContext* context) override {
    
    // get the input tensors
    const Tensor& M = context->input(1);
    Tensor init = context->input(2);
   
    // get shapes
    const TensorShape& M_shape = M.shape();    
  
    const int batch_samples = M_shape.dim_size(0);
    const int nb_incr = M_shape.dim_size(1); 
    const int paths_length = M_shape.dim_size(1)+1; 
    const int nb_diagonals = 2*paths_length -1; 
    const int paths_length_ = M_shape.dim_size(1)+2;


    // check that the incr are 3 dimensional
    DCHECK_EQ(M_shape.dims(),3);
    
    context->set_output(0, context->input(2));
    
    auto f_M = M.flat<double>();
    auto f_sol = init.flat<double>();

    
    UntruncCovRevKernelLauncher(
      f_M.data(),
      batch_samples,
      nb_incr,
      paths_length,
      paths_length_,
      nb_diagonals,
      f_sol.data()
    );

    }
  };

  REGISTER_KERNEL_BUILDER(Name("UntruncCov").Device(DEVICE_GPU), UntruncCovOp);
  REGISTER_KERNEL_BUILDER(Name("UntruncCovRev").Device(DEVICE_GPU), UntruncCovRevOp);




