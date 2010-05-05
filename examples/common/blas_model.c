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

#include "blas_model.h"
#include <starpu.h>

/*
 * As a convention, in that file, descr[0] is represented by A,
 * 				  descr[1] is B ...
 */

/*
 *	Number of flops of Gemm 
 */

double gemm_cost(starpu_buffer_descr *descr)
{
	/* C = A * B */
	uint32_t nxC, nyC, nxA;


	nxC = starpu_matrix_get_nx(descr[2].handle);
	nyC = starpu_matrix_get_ny(descr[2].handle);
	nxA = starpu_matrix_get_nx(descr[0].handle);

//	printf("nxC %d nxC %d nxA %d\n", nxC, nyC, nxA);

	double cost = ((double)nxC)*((double)nyC)*((double)nxA/1000.0f/4.11f);

//	printf("cost %e \n", cost);

	return cost;
}
