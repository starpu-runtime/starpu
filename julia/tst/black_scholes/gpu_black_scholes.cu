#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <starpu.h>

// __device__ inline double cndGPU(double d)
// {
//   const double A1 = 0.31938153f;
//   const double A2 = -0.356563782f;
//   const double A3 = 1.781477937f;
//   const double A4 = -1.821255978f;
//   const double A5 = 1.330274429f;
//   const float RSQRT2PI = 0.39894228040143267793994605993438f;

    
//   double K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    
//   double cnd = RSQRT2PI * __expf(- 0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

//     if (d > 0)
//       cnd = 1.0f - cnd;

//     return cnd;
// }

__device__ inline double cndGPU(double d)
{
  return (1.0 + erf(d/sqrt(2.0)))/2.0;
}

__global__ void gpuBlackScholesKernel(double *S, double *K, double *R, double *T, 
				      double *SIG, double *CRES, double *PRES,
				      uint32_t nxS)
{
  uint32_t i, id;
  
  id = blockIdx.x * blockDim.x + threadIdx.x;
  i = id % nxS;
  
  double sqrtT = __fdividef(1.0F, rsqrtf(T[i]));
  double d1 = (log(S[i] / K[i]) + (R[i] + SIG[i] * SIG[i] * 0.5) * T[i]) / (SIG[i] * sqrt(T[i]));  
  double d2 = (log(S[i] / K[i]) + (R[i] - SIG[i] * SIG[i] * 0.5) * T[i]) / (SIG[i] * sqrt(T[i]));
  
  CRES[i] = S[i] * (normcdf(d1)) - K[i] * exp(-R[i] * T[i]) * normcdf(d2);
  PRES[i] = -S[i] * (normcdf(-d1)) + K[i] * exp(-R[i] * T[i]) * normcdf(-d2);
}

#define THREADS_PER_BLOCK 64

extern "C" void gpu_black_scholes(void *descr[], void *args)
{
  double *S, *K, *R, *T, *SIG, *CRES, *PRES;
  uint32_t nxS;
  uint32_t nblocks;

  S = (double *) STARPU_MATRIX_GET_PTR(descr[0]);
  K = (double *) STARPU_MATRIX_GET_PTR(descr[1]);
  R = (double *) STARPU_MATRIX_GET_PTR(descr[2]);
  T = (double *) STARPU_MATRIX_GET_PTR(descr[3]);
  SIG = (double *) STARPU_MATRIX_GET_PTR(descr[4]);
  CRES = (double *) STARPU_MATRIX_GET_PTR(descr[5]);
  PRES = (double *) STARPU_MATRIX_GET_PTR(descr[6]);

  nxS = STARPU_MATRIX_GET_NX(descr[0]);

  nblocks = (nxS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  gpuBlackScholesKernel
    <<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
    >>> (S, K, R, T, SIG, CRES, PRES, nxS);
  
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}