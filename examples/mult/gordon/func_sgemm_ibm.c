/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

#include "func_sgemm_ibm.h"

#include <blas_s.h>

void func_sgemm_ibm(__attribute__ ((unused)) void **alloc,
		__attribute__ ((unused)) void **in,
		__attribute__ ((unused)) void **inout,
		__attribute__ ((unused)) void **out)
{
	/* we assume data will be in A:R,B:R,C:RW mode
 	 *  -> in[0] : describe problem
 	 *  -> in[1] : A
 	 *  -> in[2] : B
 	 *  -> inout[0] : C
 	 *
 	 *   C = AB + C
 	 *   but, being in fortran ordering, we compute
 	 *   t(C) = t(B)t(A) + t(C) instead
 	 */
	struct ibm_sgemm_block_conf *conf = in[0];
	float *A = in[1];
	float *B = in[2];
	float *C = inout[0];

	sgemm_spu(conf->m, conf->n, conf->k, B, A, C);
}
