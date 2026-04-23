/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2026-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <mkl.h>
#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <starpu_syclblas.h>
#include "sycl_gemm.h"
#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

int syclblasSgemm(unsigned nxC, unsigned nyC, unsigned nyA,
                  float alpha, const float *subA, unsigned ldA,
                  const float *subB, unsigned ldB, float beta,
                  float *subC, unsigned ldC)
{
	sycl::queue *q = starpu_sycl_get_local_stream();
	dpct::err0 syclres = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
					      *q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
					      nxC, nyC, nyA,
					      alpha, subA, ldA, subB, ldB,
					      beta, subC, ldC));
	return syclres;
}

int syclblasDgemm(unsigned nxC, unsigned nyC, unsigned nyA,
                  double alpha, const double *subA, unsigned ldA,
                  const double *subB, unsigned ldB, double beta,
                  double *subC, unsigned ldC)
{
	sycl::queue *q = starpu_sycl_get_local_stream();
	dpct::err0 syclres = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
					      *q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
					      nxC, nyC, nyA,
					      alpha, subA, ldA, subB, ldB,
					      beta, subC, ldC));
	return syclres;
}

#ifdef __cplusplus
}
#endif
