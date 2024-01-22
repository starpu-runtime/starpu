/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2012, 2014, 2019, 2021-2022  Université de Bordeaux 1
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

//! [To be included. You should update doxygen if you see this text.]
#include <starpu.h>

extern void vector_scal_cpu(void *buffers[], void *_args);
extern void vector_scal_cuda(void *buffers[], void *_args);
extern void vector_scal_opencl(void *buffers[], void *_args);

//! [Codelet To be included. You should update doxygen if you see this text.]
static struct starpu_codelet cl =
{
	.cpu_funcs = {vector_scal_cpu},
	.cuda_funcs = {vector_scal_cuda},
	.opencl_funcs = {vector_scal_opencl},

	.nbuffers = 1,
	.modes = {STARPU_RW}
};
//! [Codelet To be included. You should update doxygen if you see this text.]

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program programs;
#endif

#define    NX    2048
int main(void)
{
	float *vector;
	unsigned i;

//! [init To be included. You should update doxygen if you see this text.]
	int ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
//! [init To be included. You should update doxygen if you see this text.]

#ifdef STARPU_USE_OPENCL
	starpu_opencl_load_opencl_from_file("vector_scal_opencl_kernel.cl", &programs, NULL);
#endif

//! [alloc To be included. You should update doxygen if you see this text.]
	vector = malloc(sizeof(vector[0]) * NX);
	for (i = 0; i < NX; i++)
		vector[i] = 1.0f;
	fprintf(stderr, "BEFORE : First element was %f\n", vector[0]);
//! [alloc To be included. You should update doxygen if you see this text.]

//! [register To be included. You should update doxygen if you see this text.]
	starpu_data_handle_t vector_handle;
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));
//! [register To be included. You should update doxygen if you see this text.]

//! [task_insert To be included. You should update doxygen if you see this text.]
	float factor = 3.14;
	ret = starpu_task_insert(&cl,
				 STARPU_RW, vector_handle,
				 STARPU_VALUE, &factor, sizeof(factor),
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
//! [task_insert To be included. You should update doxygen if you see this text.]

//! [wait To be included. You should update doxygen if you see this text.]
	starpu_task_wait_for_all();
	starpu_data_unregister(vector_handle);
//! [wait To be included. You should update doxygen if you see this text.]

	fprintf(stderr, "AFTER First element is %f\n", vector[0]);
	free(vector);

#ifdef STARPU_USE_OPENCL
	starpu_opencl_unload_opencl(&programs);
#endif

//! [shutdown To be included. You should update doxygen if you see this text.]
	starpu_shutdown();
//! [shutdown To be included. You should update doxygen if you see this text.]
	return 0;
}
//! [To be included. You should update doxygen if you see this text.]
