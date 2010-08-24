/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#include "dw_mult.h"

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_GEMM_STR(name)  xstr(STARPU_GEMM(name))

TYPE *A, *B, *C;
starpu_data_handle A_handle, B_handle, C_handle;

/*
 * That program should compute C = A * B 
 * 
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)

              |---------------|
            z |       B       |
              |---------------|
       z              x
     |----|   |---------------|
     |    |   |               |
     |    |   |               |
     | A  | y |       C       |
     |    |   |               |
     |    |   |               |
     |----|   |---------------|

 */

static void check_output(void)
{
	/* check results */
	/* compute C = C - AB */

	CPU_GEMM("N", "N", ydim, xdim, zdim, (TYPE)-1.0, A, ydim, B, zdim, (TYPE)1.0f, C, ydim);
		
	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(xdim*ydim, C, 1);

	int max;
	max = CPU_IAMAX(xdim*ydim, C, 1);

	fprintf(stderr, "Avg error : %e\n", err/(xdim*ydim));
	fprintf(stderr, "Max error : %e\n", C[max]);
}

void callback_func(void *arg)
{
	/* do some accounting */
	int id = starpu_worker_get_id();
	flop_per_worker[id] += BLAS3_FLOP(conf.m, conf.n, conf.k);
	ls_per_worker[id] += BLAS3_LS(conf.m, conf.n, conf.k);
}

static void init_problem_data(void)
{
	unsigned i,j;

#ifdef STARPU_USE_CUDA
	if (pin) {
		starpu_data_malloc_pinned_if_possible((void **)&A, zdim*ydim*sizeof(TYPE));
		starpu_data_malloc_pinned_if_possible((void **)&B, xdim*zdim*sizeof(TYPE));
		starpu_data_malloc_pinned_if_possible((void **)&C, xdim*ydim*sizeof(TYPE));
	} else
#endif
	{
#ifdef STARPU_HAVE_POSIX_MEMALIGN
		posix_memalign((void **)&A, 4096, zdim*ydim*sizeof(TYPE));
		posix_memalign((void **)&B, 4096, xdim*zdim*sizeof(TYPE));
		posix_memalign((void **)&C, 4096, xdim*ydim*sizeof(TYPE));
#else
		A = malloc(zdim*ydim*sizeof(TYPE));
		B = malloc(xdim*zdim*sizeof(TYPE));
		C = malloc(xdim*ydim*sizeof(TYPE));
#endif
	}

	/* fill the A and B matrices */
	if (norandom) {
		for (j=0; j < ydim; j++) {
			for (i=0; i < zdim; i++) {
				A[j+i*ydim] = (TYPE)(i);
			}
		}
	
		for (j=0; j < zdim; j++) {
			for (i=0; i < xdim; i++) {
				B[j+i*zdim] = (TYPE)(j);
			}
		}
	} 
	else {
		for (j=0; j < ydim; j++) {
			for (i=0; i < zdim; i++) {
				A[j+i*ydim] = (TYPE)(starpu_drand48());
			}
		}
	
		for (j=0; j < zdim; j++) {
			for (i=0; i < xdim; i++) {
				B[j+i*zdim] = (TYPE)(starpu_drand48());
			}
		}
	}

	for (j=0; j < ydim; j++) {
		for (i=0; i < xdim; i++) {
			C[j+i*ydim] = (TYPE)(0);
		}
	}

	/* display memory consumption */
	fprintf(stderr, "Total memory : %ld MB\n",
		( ydim*zdim*sizeof(TYPE)
		+ zdim*xdim*sizeof(TYPE)
		+ ydim*xdim*sizeof(TYPE) )/(1024*1024));

}

static void partition_mult_data(void)
{
	starpu_matrix_data_register(&A_handle, 0, (uintptr_t)A, 
		ydim, ydim, zdim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, 0, (uintptr_t)B, 
		zdim, zdim, xdim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, 0, (uintptr_t)C, 
		ydim, ydim, xdim, sizeof(TYPE));

	starpu_data_set_wt_mask(C_handle, 1<<0);

	conf.k = zdim;
	conf.m = ydim/nslicesy;
	conf.n = xdim/nslicesx;

	struct starpu_data_filter f;
	f.filter_func = starpu_vertical_block_filter_func;
	f.nchildren = nslicesx;
	f.get_nchildren = NULL;
	f.get_child_ops = NULL;
		
	struct starpu_data_filter f2;
	f2.filter_func = starpu_block_filter_func;
	f2.nchildren = nslicesy;
	f2.get_nchildren = NULL;
	f2.get_child_ops = NULL;
		
	starpu_data_partition(B_handle, &f);
	starpu_data_partition(A_handle, &f2);

	starpu_data_map_filters(C_handle, 2, &f, &f2);
}

static void unpartition_mult_data(void)
{
	starpu_data_unpartition(C_handle, 0);

	starpu_data_unregister(C_handle);
}

static struct starpu_perfmodel_t gemm_model = {
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_GEMM_STR(gemm_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_GEMM_STR(gemm_goto)
#else
	.symbol = STARPU_GEMM_STR(gemm)
#endif
};

static starpu_codelet cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = STARPU_GEMM(cpu_mult),
#ifdef STARPU_USE_CUDA
	.cuda_func = STARPU_GEMM(cublas_mult),
#endif
	.model = &gemm_model,
	.nbuffers = 3
};

static void launch_codelets(void)
{
	/* partition the work into slices */
	unsigned taskx, tasky;

	for (taskx = 0; taskx < nslicesx; taskx++) 
	{
		for (tasky = 0; tasky < nslicesy; tasky++)
		{
			/* A B[task] = C[task] */
			struct starpu_task *task = starpu_task_create();

			task->cl = &cl;
			task->cl_arg = &conf;
			task->cl_arg_size = sizeof(struct block_conf);

			/* we have a callback to do some accounting */
			task->callback_func = callback_func;
			task->callback_arg = NULL;

			task->buffers[0].handle = starpu_data_get_sub_data(A_handle, 1, tasky);
			task->buffers[0].mode = STARPU_R;
			task->buffers[1].handle = starpu_data_get_sub_data(B_handle, 1, taskx);
			task->buffers[1].mode = STARPU_R;
			task->buffers[2].handle = starpu_data_get_sub_data(C_handle, 2, taskx, tasky);
			task->buffers[2].mode = STARPU_RW;

			starpu_task_submit(task);
		}
	}
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	starpu_init(NULL);
	starpu_helper_cublas_init();

	init_problem_data();

	gettimeofday(&start, NULL);

	partition_mult_data();

	launch_codelets();
	starpu_task_wait_for_all();

	gettimeofday(&end, NULL);
	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 +
					(end.tv_usec - start.tv_usec));
	display_stats(timing);

	unpartition_mult_data();
	
	if (check)
		check_output();
	
	starpu_helper_cublas_shutdown();
	starpu_shutdown();

	return 0;
}
