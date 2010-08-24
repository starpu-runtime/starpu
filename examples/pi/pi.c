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

#include "SobolQRNG/sobol.h"
#include "pi.h"
#include <sys/time.h>

#ifdef STARPU_USE_CUDA
void cuda_kernel(void **descr, void *cl_arg);
#endif

/* default value */
static unsigned ntasks = 1024;

static void cpu_kernel(void *descr[], void *cl_arg)
{
	unsigned *directions = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned nx = NSHOT_PER_TASK;

	TYPE *random_numbers = malloc(2*nx*sizeof(TYPE));
	sobolCPU(2*nx/n_dimensions, n_dimensions, directions, random_numbers);

	TYPE *random_numbers_x = &random_numbers[0];
	TYPE *random_numbers_y = &random_numbers[nx];

	unsigned current_cnt = 0;

	unsigned i;
	for (i = 0; i < nx; i++)
	{
		TYPE x = random_numbers_x[i];
		TYPE y = random_numbers_y[i];

		TYPE dist = (x*x + y*y);

		unsigned success = (dist <= 1.0);
		current_cnt += success;
	}

	unsigned *cnt = (unsigned *)STARPU_VECTOR_GET_PTR(descr[1]);
	*cnt = current_cnt;

	free(random_numbers);
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-ntasks") == 0) {
			char *argptr;
			ntasks = strtol(argv[++i], &argptr, 10);
		}
	}
}

int main(int argc, char **argv)
{
	unsigned i;

	parse_args(argc, argv);

	starpu_init(NULL);

	/* Initialize the random number generator */
	unsigned *sobol_qrng_directions = malloc(n_dimensions*n_directions*sizeof(unsigned));
	STARPU_ASSERT(sobol_qrng_directions);

	initSobolDirectionVectors(n_dimensions, sobol_qrng_directions);

	/* Any worker may use that array now */
	starpu_data_handle sobol_qrng_direction_handle;
	starpu_vector_data_register(&sobol_qrng_direction_handle, 0,
		(uintptr_t)sobol_qrng_directions, n_dimensions*n_directions, sizeof(unsigned));

	unsigned *cnt_array = malloc(ntasks*sizeof(unsigned));
	STARPU_ASSERT(cnt_array);
	starpu_data_handle cnt_array_handle;
	starpu_vector_data_register(&cnt_array_handle, 0, (uintptr_t)cnt_array, ntasks, sizeof(unsigned));

	/* Use a write-through policy : when the data is modified on an
	 * accelerator, we know that it will only be modified once and be
	 * accessed by the CPU later on */
	starpu_data_set_wt_mask(cnt_array_handle, (1<<0));

	struct starpu_data_filter f = {
		.filter_func = starpu_block_filter_func_vector,
		.nchildren = ntasks,
		.get_nchildren = NULL,
		.get_child_ops = NULL
	};
	
	starpu_data_partition(cnt_array_handle, &f);

	static struct starpu_perfmodel_t model = {
		.type = STARPU_HISTORY_BASED,
		.symbol = "monte_carlo_pi"
	};

	struct starpu_codelet_t cl = {
		.where = STARPU_CPU|STARPU_CUDA,
		.cpu_func = cpu_kernel,
#ifdef STARPU_USE_CUDA
		.cuda_func = cuda_kernel,
#endif
		.nbuffers = 2,
		.model = &model
	};

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;

		STARPU_ASSERT(starpu_data_get_sub_data(cnt_array_handle, 1, i));

		task->buffers[0].handle = sobol_qrng_direction_handle;
		task->buffers[0].mode   = STARPU_R;
		task->buffers[1].handle = starpu_data_get_sub_data(cnt_array_handle, 1, i);
		task->buffers[1].mode   = STARPU_W;

		int ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_task_wait_for_all();

	/* Get the cnt_array back in main memory */
	starpu_data_unpartition(cnt_array_handle, 0);
	starpu_data_acquire(cnt_array_handle, STARPU_RW);

	/* Count the total number of entries */
	unsigned long total_cnt = 0;
	for (i = 0; i < ntasks; i++)
		total_cnt += cnt_array[i];

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	unsigned long total_shot_cnt = ntasks * NSHOT_PER_TASK;

	/* Total surface : Pi * r^ 2 = Pi*1^2, total square surface : 2^2 = 4, probability to impact the disk: pi/4 */
	fprintf(stderr, "Pi approximation : %f (%ld / %ld)\n", ((TYPE)total_cnt*4)/(total_shot_cnt), total_cnt, total_shot_cnt);
	fprintf(stderr, "Total time : %f ms\n", timing/1000.0);
	fprintf(stderr, "Speed : %f GShot/s\n", total_shot_cnt/(1e3*timing));

	starpu_data_release(cnt_array_handle);

	starpu_display_codelet_stats(&cl);

	starpu_shutdown();

	return 0;
}
