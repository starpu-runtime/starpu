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

#include "incrementer_cuda.h"

extern "C" __global__ 
void cuda_incrementer(float * tab, uint32_t nx, uint32_t pad1, float *unity, uint32_t nx2, uint32_t pad2)
{
	tab[0] = tab[0] + unity[0];
	tab[1] = tab[1] + unity[1];
	tab[2] = tab[2] + unity[2];
	
	return;
}
