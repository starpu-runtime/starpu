/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_config.h>

#if defined(STARPU_HAVE_FFTW) && !defined(__CUDACC__) 
#include <complex.h>
#include <fftw3.h>
#endif

#ifdef STARPU_USE_CUDA
#include <cufft.h>
#endif

#undef  DOUBLE
#define FLOAT

typedef float real;
#if defined(STARPU_HAVE_FFTW) && !defined(__CUDACC__) 
typedef fftwf_complex _fftw_complex;
typedef fftwf_plan _fftw_plan;
#endif
#ifdef STARPU_USE_CUDA
typedef cuComplex _cuComplex;
typedef cufftComplex _cufftComplex;
#define _cufftExecC2C cufftExecC2C
#define _cufftExecR2C cufftExecR2C
#define _cufftExecC2R cufftExecC2R
#define _CUFFT_C2C CUFFT_C2C
#define _CUFFT_R2C CUFFT_R2C
#define _CUFFT_C2R CUFFT_C2R
#define _cuCmul(x,y) cuCmulf(x,y)
#endif
#define STARPUFFT(name) starpufftf_##name
#define _FFTW(name) fftwf_##name

#ifdef STARPU_USE_CUDA
void STARPUFFT(report_error)(const char *func, const char *file, int line, cufftResult status);
#endif /* !STARPU_USE_CUDA */

#define TYPE "f"
