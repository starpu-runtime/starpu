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

static __global__ void cuda_variable(float * tab)
{
	*tab += 1.0;
	return;
}

extern "C" void cuda_codelet(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	float *val = (float *)STARPU_VARIABLE_GET_PTR(descr[0]);

	cuda_variable<<<1,1>>>(val);
}
