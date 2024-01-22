/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2021  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
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

//! [starpu scal code To be included. You should update doxygen if you see this text.]
//! [Prototype To be included. You should update doxygen if you see this text.]
void vector_scal_cpu(void *buffers[], void *cl_arg)
{
//! [Prototype To be included. You should update doxygen if you see this text.]
//! [Extract To be included. You should update doxygen if you see this text.]
	struct starpu_vector_interface *vector = buffers[0];
	float *val = (float *)STARPU_VECTOR_GET_PTR(vector);
	unsigned n = STARPU_VECTOR_GET_NX(vector);
//! [Extract To be included. You should update doxygen if you see this text.]

//! [Unpack To be included. You should update doxygen if you see this text.]
	float factor;
	starpu_codelet_unpack_args(cl_arg, &factor);
//! [Unpack To be included. You should update doxygen if you see this text.]

//! [Compute To be included. You should update doxygen if you see this text.]
	unsigned i;
	for (i = 0; i < n; i++)
		val[i] *= factor;
//! [Compute To be included. You should update doxygen if you see this text.]
}
//! [starpu scal code To be included. You should update doxygen if you see this text.]
