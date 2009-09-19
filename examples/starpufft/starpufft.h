/*
 * StarPU
 * Copyright (C) INRIA 2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdio.h>
#include <complex.h>
#include <starpu.h>

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
starpufft(plan) starpufft(plan_dft_r2c_1d)(int n, unsigned flags); \
starpufft(plan) starpufft(plan_dft_c2r_1d)(int n, unsigned flags); \
\
void *starpufft(malloc)(size_t n); \
void starpufft(free)(void *p); \
\
void starpufft(execute)(starpufft(plan) p, void *in, void *out); \
starpu_tag_t starpufft(start)(starpufft(plan) p, void *in, void *out); \
\
void starpufft(destroy_plan)(starpufft(plan) p); \
\
void starpufft(startstats)(void); \
void starpufft(stopstats)(void); \
void starpufft(showstats)(FILE *out);

__STARPUFFT_INTERFACE(__STARPUFFT, double)
__STARPUFFT_INTERFACE(__STARPUFFTF, float)
__STARPUFFT_INTERFACE(__STARPUFFTL, long double)

int starpufft_last_plan_number;
