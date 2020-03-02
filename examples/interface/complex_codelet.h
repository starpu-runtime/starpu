/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

/* Dumb performance model for simgrid */
static double complex_cost_function(struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;
	return 0.000001;
}

static struct starpu_perfmodel complex_model =
{
	.type = STARPU_COMMON,
	.cost_function = complex_cost_function,
	.symbol = "complex"
};

void compare_complex_codelet(void *descr[], void *_args)
{
	int nx1 = STARPU_COMPLEX_GET_NX(descr[0]);
	double *real1 = STARPU_COMPLEX_GET_REAL(descr[0]);
	double *imaginary1 = STARPU_COMPLEX_GET_IMAGINARY(descr[0]);

	int nx2 = STARPU_COMPLEX_GET_NX(descr[1]);
	double *real2 = STARPU_COMPLEX_GET_REAL(descr[1]);
	double *imaginary2 = STARPU_COMPLEX_GET_IMAGINARY(descr[1]);

	int *compare;

	starpu_codelet_unpack_args(_args, &compare);
	*compare = (nx1 == nx2);
	if (nx1 == nx2)
	{
		int i;
		for(i=0 ; i<nx1 ; i++)
		{
			if (real1[i] != real2[i] || imaginary1[i] != imaginary2[i])
			{
				*compare = 0;
				break;
			}
		}
	}
}

struct starpu_codelet cl_compare =
{
	.cpu_funcs = {compare_complex_codelet},
	/* dereferencing compare won't work on MIC */
	/* .cpu_funcs_name = {"compare_complex_codelet"}, */
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_R},
	.name = "cl_compare",
	.model = &complex_model
};

void display_complex_codelet(void *descr[], void *_args)
{
	int nx = STARPU_COMPLEX_GET_NX(descr[0]);
	double *real = STARPU_COMPLEX_GET_REAL(descr[0]);
	double *imaginary = STARPU_COMPLEX_GET_IMAGINARY(descr[0]);
	int i;
	char msg[100];

	if (_args)
		starpu_codelet_unpack_args(_args, &msg);

	for(i=0 ; i<nx ; i++)
	{
		FPRINTF(stderr, "[%s] Complex[%d] = %3.2f + %3.2f i\n", _args?msg:NULL, i, real[i], imaginary[i]);
	}
}

struct starpu_codelet cl_display =
{
	.cpu_funcs = {display_complex_codelet},
	.cpu_funcs_name = {"display_complex_codelet"},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.name = "cl_display",
	.model = &complex_model
};

#endif /* __COMPLEX_CODELET_H */
