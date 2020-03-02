/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

/*
 * This examplifies the use of the shadow filter: a source "vector" of NX
 * elements (plus 2*SHADOW wrap-around elements) is partitioned into vectors
 * with some shadowing, and these are copied into a destination "vector2" of
 * NRPARTS*(NX/NPARTS+2*SHADOW) elements, partitioned in the traditionnal way,
 * thus showing how shadowing shows up.
 *
 * For instance, with NX=8, SHADOW=1, and NPARTS=4:
 *
 * vector
 * x0 x1 x2 x3 x4 x5 x6 x7 x8 x9
 *
 * is partitioned into 4 pieces:
 *
 * x0 x1 x2 x3
 *       x2 x3 x4 x5
 *             x4 x5 x6 x7
 *                   x6 x7 x8 x9
 *
 * which are copied into the 4 destination subparts of vector2, thus getting in
 * the end:
 *
 * x0 x1 x2 x3 x2 x3 x4 x5 x4 x5 x6 x7 x6 x7 x8 x9
 */

#include <starpu.h>

/* Shadow width */
#define SHADOW 2
#define NX    30
#define PARTS 3

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
        unsigned i;

        /* length of the shadowed source vector */
        unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
        /* local copy of the shadowed source vector pointer */
        int *val = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

        /* length of the destination vector */
        unsigned n2 = STARPU_VECTOR_GET_NX(buffers[1]);
        /* local copy of the destination vector pointer */
        int *val2 = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);

	/* If things go right, sizes should match */
	STARPU_ASSERT(n == n2);
	for (i = 0; i < n; i++)
		val2[i] = val[i];
}

#ifdef STARPU_USE_CUDA
void cuda_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
        /* length of the shadowed source vector */
        unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
        /* local copy of the shadowed source vector pointer */
        int *val = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

        /* length of the destination vector */
        unsigned n2 = STARPU_VECTOR_GET_NX(buffers[1]);
        /* local copy of the destination vector pointer */
        int *val2 = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);

	/* If things go right, sizes should match */
	STARPU_ASSERT(n == n2);
	cudaMemcpyAsync(val2, val, n*sizeof(*val), cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
}
#endif

int main(void)
{
	unsigned j;
        int vector[NX + 2*SHADOW];
        int vector2[NX + PARTS*2*SHADOW];
	starpu_data_handle_t handle, handle2;
	int ret, i;

        struct starpu_codelet cl =
	{
                .cpu_funcs = {cpu_func},
                .cpu_funcs_name = {"cpu_func"},
#ifdef STARPU_USE_CUDA
                .cuda_funcs = {cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
                .nbuffers = 2,
		.modes = {STARPU_R, STARPU_W}
        };

        for(i=0 ; i<NX ; i++) vector[SHADOW+i] = i;
	for(i=0 ; i<SHADOW ; i++) vector[i] = vector[i+NX];
	for(i=0 ; i<SHADOW ; i++) vector[SHADOW+NX+i] = vector[SHADOW+i];
        FPRINTF(stderr,"IN  Vector: ");
        for(i=0 ; i<NX + 2*SHADOW ; i++) FPRINTF(stderr, "%5d ", vector[i]);
        FPRINTF(stderr,"\n");

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Declare source vector to StarPU */
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX + 2*SHADOW, sizeof(vector[0]));

	/* Declare destination vector to StarPU */
	starpu_vector_data_register(&handle2, STARPU_MAIN_RAM, (uintptr_t)vector2, NX + PARTS*2*SHADOW, sizeof(vector[0]));

        /* Partition the source vector in PARTS sub-vectors with shadows */
	/* NOTE: the resulting handles should only be used in read-only mode,
	 * as StarPU will not know how the overlapping parts would have to be
	 * combined. */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block_shadow,
		.nchildren = PARTS,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOW /* Shadow width */
	};
	starpu_data_partition(handle, &f);

        /* Partition the destination vector in PARTS sub-vectors */
	struct starpu_data_filter f2 =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = PARTS,
	};
	starpu_data_partition(handle2, &f2);

        /* Submit a task on each sub-vector */
	for (i=0; i<starpu_data_get_nb_children(handle); i++)
	{
                starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 1, i);
                starpu_data_handle_t sub_handle2 = starpu_data_get_sub_data(handle2, 1, i);
                struct starpu_task *task = starpu_task_create();

		task->handles[0] = sub_handle;
		task->handles[1] = sub_handle2;
                task->cl = &cl;
                task->synchronous = 1;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(handle2, STARPU_MAIN_RAM);
        starpu_data_unregister(handle);
        starpu_data_unregister(handle2);
	starpu_shutdown();

        FPRINTF(stderr,"OUT Vector: ");
        for(i=0 ; i<NX + PARTS*2*SHADOW ; i++) FPRINTF(stderr, "%5d ", vector2[i]);
        FPRINTF(stderr,"\n");
	for(i=0 ; i<PARTS ; i++)
		for (j=0 ; j<NX/PARTS ; j++)
			STARPU_ASSERT(vector2[i*(NX/PARTS+2*SHADOW)+j] == vector[i*(NX/PARTS)+j]);

	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
