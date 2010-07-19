/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

extern void cuda_func(void *buffers[], void *cl_arg);
extern void opencl_func(void *buffers[], void *cl_arg);

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

void print_block(int *block, int nx, int ny, int nz, unsigned ldy, unsigned ldz)
{
        int i, j, k;
        fprintf(stderr, "block=%p nx=%d ny=%d nz=%d ldy=%d ldz=%d\n", block, nx, ny, nz, ldy, ldz);
        for(k=0 ; k<nz ; k++) {
                for(j=0 ; j<ny ; j++) {
                        for(i=0 ; i<nx ; i++) {
                                fprintf(stderr, "%2d ", block[(k*ldz)+(j*ldy)+i]);
                        }
                        fprintf(stderr,"\n");
                }
                fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
}

void print_data(starpu_data_handle block_handle)
{
	int *block = starpu_block_get_local_ptr(block_handle);
	int nx = starpu_block_get_nx(block_handle);
	int ny = starpu_block_get_ny(block_handle);
	int nz = starpu_block_get_nz(block_handle);
	unsigned ldy = starpu_block_get_local_ldy(block_handle);
	unsigned ldz = starpu_block_get_local_ldz(block_handle);

        print_block(block, nx, ny, nz, ldy, ldz);
}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program codelet;
#endif

int main(int argc, char **argv)
{
        int *block,n=0;
        int i, j, k;

        block = (int*)malloc(NX*NY*NZ*sizeof(block[0]));
        assert(block);
        for(k=0 ; k<NZ ; k++) {
                for(j=0 ; j<NY ; j++) {
                        for(i=0 ; i<NX ; i++) {
                                block[(k*NX*NY)+(j*NX)+i] = n++;
                        }
                }
        }

	starpu_data_handle handle;
	starpu_codelet cl =
	{
                .where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
                .cpu_func = cpu_func,
                .cuda_func = cuda_func,
                .opencl_func = opencl_func,
		.nbuffers = 1
	};
        starpu_init(NULL);

#ifdef STARPU_USE_OPENCL
        starpu_opencl_load_opencl_from_file("examples/filters/fblock_opencl_codelet.cl", &codelet);
#endif

        /* Declare data to StarPU */
        starpu_block_data_register(&handle, 0, (uintptr_t)block, NX, NX*NY, NX, NY, NZ, sizeof(int));
        fprintf(stderr, "IN  Block\n");
        print_data(handle);

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

        for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
        {
                starpu_data_handle sblock = starpu_data_get_sub_data(handle, 1, i);
                fprintf(stderr, "Sub block %d\n", i);
                print_data(sblock);
        }

        /* Submit a task on each sub-block */
        for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
        {
                int ret,multiplier=i;
                struct starpu_task *task = starpu_task_create();

                fprintf(stderr,"Dealing with sub-block %d\n", i);
                task->cl = &cl;
                task->synchronous = 1;
                task->callback_func = NULL;
                task->buffers[0].handle = starpu_data_get_sub_data(handle, 1, i);
                task->buffers[0].mode = STARPU_RW;
                task->cl_arg = &multiplier;

                ret = starpu_task_submit(task);
                if (ret) {
                        fprintf(stderr, "Error when submitting task\n");
                        exit(ret);
                }
        }

        /* Unpartition the data, unregister it from StarPU and shutdown */
        starpu_data_unpartition(handle, 0);
        print_data(handle);
        starpu_data_unregister(handle);

        /* Print result block */
        fprintf(stderr, "OUT Block\n");
        print_block(block, NX, NY, NZ, NX, NX*NY);

	starpu_shutdown();
}
