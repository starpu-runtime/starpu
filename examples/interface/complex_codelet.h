/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include "complex_interface.h"

#ifndef __COMPLEX_CODELET_H
#define __COMPLEX_CODELET_H


void display_complex_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	int nx = STARPU_COMPLEX_GET_NX(descr[0]);
	double *real = STARPU_COMPLEX_GET_REAL(descr[0]);
	double *imaginary = STARPU_COMPLEX_GET_IMAGINARY(descr[0]);
	int i;

	for(i=0 ; i<nx ; i++)
	{
		fprintf(stderr, "Complex[%d] = %3.2f + %3.2f i\n", i, real[i], imaginary[i]);
	}
}

struct starpu_codelet cl_display =
{
	.cpu_funcs = {display_complex_codelet, NULL},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

#endif /* __COMPLEX_CODELET_H */
