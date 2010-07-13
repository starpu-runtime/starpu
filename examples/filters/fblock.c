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
#include <starpu_opencl.h>

#define NX    5
#define NY    4
#define NZ    3
#define PARTS 2

void cpu_func(void *buffers[], void *cl_arg)
{
        unsigned i, j, k;
        int *factor = cl_arg;

	int *block = (int *)STARPU_GET_BLOCK_PTR(buffers[0]);
	int nx = (int)STARPU_GET_BLOCK_NX(buffers[0]);
	int ny = (int)STARPU_GET_BLOCK_NY(buffers[0]);
	int nz = (int)STARPU_GET_BLOCK_NZ(buffers[0]);
        unsigned ldy = STARPU_GET_BLOCK_LDY(buffers[0]);
        unsigned ldz = STARPU_GET_BLOCK_LDZ(buffers[0]);

        for(k=0; k<nz ; k++) {
                for(j=0; j<ny ; j++) {
                        for(i=0; i<nx ; i++)
                                block[(k*ldz)+(j*ldy)+i] = *factor;
                }
        }
}

int main(int argc, char **argv)
{
        int *block,n=0;
        int i, j, k;

        block = (int*)malloc(NX*NY*NZ*sizeof(block[0]));
        assert(block);
        fprintf(stderr, "IN  Block\n");
        for(k=0 ; k<NZ ; k++) {
                for(j=0 ; j<NY ; j++) {
                        for(i=0 ; i<NX ; i++) {
                                block[(k*NY)+(j*NX)+i] = n++;
                                fprintf(stderr, "%2d ", block[(k*NY)+(j*NX)+i]);
                        }
                        fprintf(stderr,"\n");
                }
                fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");

	starpu_data_handle handle;
	starpu_codelet cl =
	{
		.where = STARPU_CPU,
                .cpu_func = cpu_func,
		.nbuffers = 1
	};
        starpu_init(NULL);

	/* Declare data to StarPU */
	starpu_block_data_register(&handle, 0, (uintptr_t)block, NX, NX*NY, NX, NY, NZ, sizeof(int));

        /* Partition the block in PARTS sub-blocks */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_func_block,
		.nchildren = PARTS,
		.get_nchildren = NULL,
		.get_child_ops = NULL
	};
	starpu_data_partition(handle, &f);

        fprintf(stderr,"Nb of partitions : %d\n",starpu_data_get_nb_children(handle));

        /* Submit a task on each sub-block */
        for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
        {
                int multiplier=i;
                struct starpu_task *task = starpu_task_create();

                fprintf(stderr,"Dealing with sub-block %d\n", i);
                task->cl = &cl;
                task->synchronous = 1;
                task->callback_func = NULL;
                task->buffers[0].handle = starpu_data_get_sub_data(handle, 1, i);
                task->buffers[0].mode = STARPU_RW;
                task->cl_arg = &multiplier;

                starpu_task_submit(task);
        }

        /* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, 0);
        starpu_data_unregister(handle);
	starpu_shutdown();

        /* Print result block */
        fprintf(stderr, "OUT Block\n");
        for(k=0 ; k<NZ ; k++) {
                for(j=0 ; j<NY ; j++) {
                        for(i=0 ; i<NX ; i++) {
                                fprintf(stderr, "%2d ", block[(k*NY)+(j*NX)+i]);
                        }
                        fprintf(stderr,"\n");
                }
                fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
}
