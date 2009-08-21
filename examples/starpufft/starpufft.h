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

#define STARPUFFT_FORWARD -1
#define STARPUFFT_INVERSE 1

#if defined(_Complex_I) && defined(complex) && defined(I)
typedef float _Complex starpufftf_complex;
#else
typedef float starpufftf_complex[2];
#endif

typedef struct starpufftf_plan *starpufftf_plan;

starpufftf_plan starpufftf_plan_dft_1d(int n, int sign, unsigned flags);
starpufftf_plan starpufftf_plan_dft_r2c_1d(int n, unsigned flags);
starpufftf_plan starpufftf_plan_dft_c2r_1d(int n, unsigned flags);

void *starpufftf_malloc(size_t n);
void starpufftf_free(void *p);

void starpufftf_execute(starpufftf_plan p, void *in, void *out);

void starpufftf_destroy_plan(starpufftf_plan p);

void starpufftf_startstats(void);
void starpufftf_stopstats(void);
void starpufftf_showstats(FILE *out);
