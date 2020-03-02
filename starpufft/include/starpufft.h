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

// The documentation for this file is in doc/doxygen/chapters/api/fft_support.doxy

#ifndef __STARPU_FFT_H__
#define __STARPU_FFT_H__

#include <stdio.h>
#include <complex.h>
#include <starpu.h>
#ifdef STARPU_USE_CUDA
#include <cufft.h>
#define STARPU_CUFFT_REPORT_ERROR(status) STARPUFFT(report_error)(__starpu_func__, __FILE__, __LINE__, status)
#endif /* !STARPU_USE_CUDA */

#define STARPUFFT_FORWARD -1
#define STARPUFFT_INVERSE 1

#define __STARPUFFT(name) starpufft_##name
#define __STARPUFFTF(name) starpufftf_##name
#define __STARPUFFTL(name) starpufftl_##name

#define __STARPUFFT_INTERFACE(starpufft,real) \
typedef real _Complex starpufft(complex); \
\
typedef struct starpufft(plan) *starpufft(plan); \
\
starpufft(plan) starpufft(plan_dft_1d)(int n, int sign, unsigned flags); \
starpufft(plan) starpufft(plan_dft_2d)(int n, int m, int sign, unsigned flags); \
starpufft(plan) starpufft(plan_dft_3d)(int n, int m, int p, int sign, unsigned flags); \
starpufft(plan) starpufft(plan_dft_r2c_1d)(int n, unsigned flags); \
starpufft(plan) starpufft(plan_dft_c2r_1d)(int n, unsigned flags); \
\
void *starpufft(malloc)(size_t n); \
void starpufft(free)(void *p); \
\
int starpufft(execute)(starpufft(plan) p, void *in, void *out); \
struct starpu_task *starpufft(start)(starpufft(plan) p, void *in, void *out); \
\
int starpufft(execute_handle)(starpufft(plan) p, starpu_data_handle_t in, starpu_data_handle_t out); \
struct starpu_task *starpufft(start_handle)(starpufft(plan) p, starpu_data_handle_t in, starpu_data_handle_t out); \
\
void starpufft(cleanup)(starpufft(plan) p); \
void starpufft(destroy_plan)(starpufft(plan) p); \
\
void starpufft(startstats)(void); \
void starpufft(stopstats)(void); \
void starpufft(showstats)(FILE *out);

__STARPUFFT_INTERFACE(__STARPUFFT, double)
__STARPUFFT_INTERFACE(__STARPUFFTF, float)
__STARPUFFT_INTERFACE(__STARPUFFTL, long double)

/* Internal use */
extern int starpufft_last_plan_number;

#endif // __STARPU_FFT_H__
