/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <starpu.h>

static __global__ void cuda_block(float *block, int nx, int ny, int nz, float *multiplier)
{
        int i;
        for(i=0 ; i<nx*ny*nz ; i++) block[i] *= *multiplier;
}

extern "C" void cuda_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
        float *block = (float *)STARPU_GET_BLOCK_PTR(descr[0]);
	int nx = STARPU_GET_BLOCK_NX(descr[0]);
	int ny = STARPU_GET_BLOCK_NY(descr[0]);
	int nz = STARPU_GET_BLOCK_NZ(descr[0]);
        float *multiplier = (float *)STARPU_GET_VARIABLE_PTR(descr[1]);

        cuda_block<<<1,1>>>(block, nx, ny, nz, multiplier);
}
