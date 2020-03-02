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
 * This computes Pi by using drawing random coordinates (thanks to the sobol
 * generator) and check whether they fall within one quarter of a circle.  The
 * proportion gives an approximation of Pi. For each task, we draw a number of
 * coordinates, and we gather the number of successful draws.
 *
 * TODO: use curandGenerateUniform instead of the sobol generator, like pi_redux.c does
 */

#include "SobolQRNG/sobol.h"
#include "SobolQRNG/sobol_gold.h"
#include "pi.h"

#ifdef STARPU_USE_CUDA
void cuda_kernel(void **descr, void *cl_arg);
#endif

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

/* default value */
static unsigned ntasks = 1024;

static unsigned long long nshot_per_task = 16*1024*1024ULL;

void cpu_kernel(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	unsigned *directions = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned nx = nshot_per_task;

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

/* The amount of work does not depend on the data size at all :) */
static size_t size_base(struct starpu_task *task, unsigned nimpl)
{
	(void)task;
	(void)nimpl;
	return nshot_per_task;
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-ntasks") == 0)
		{
			char *argptr;
			ntasks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nshot") == 0)
		{
			char *argptr;
			nshot_per_task = strtol(argv[++i], &argptr, 10);
		}
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			fprintf(stderr,"Usage: %s [options...]\n", argv[0]);
			fprintf(stderr,"\n");
			fprintf(stderr,"Options:\n");
			fprintf(stderr,"-ntasks <n>		select the number of tasks\n");
			fprintf(stderr,"-nshot <n>		select the number of shot per task\n");
			exit(0);
		}
	}
}

static struct starpu_perfmodel model =
{
	.type = STARPU_HISTORY_BASED,
	.size_base = size_base,
	.symbol = "monte_carlo_pi"
};

static struct starpu_codelet pi_cl =
{
	.cpu_funcs = {cpu_kernel},
	.cpu_funcs_name = {"cpu_kernel"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_kernel},
#endif
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_W},
	.model = &model
};

int main(int argc, char **argv)
{
	unsigned i;
	int ret;

	parse_args(argc, argv);

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Initialize the random number generator */
	unsigned *sobol_qrng_directions = malloc(n_dimensions*n_directions*sizeof(unsigned));
	STARPU_ASSERT(sobol_qrng_directions);

	initSobolDirectionVectors(n_dimensions, sobol_qrng_directions);

	/* Any worker may use that array now */
	starpu_data_handle_t sobol_qrng_direction_handle;
	starpu_vector_data_register(&sobol_qrng_direction_handle, STARPU_MAIN_RAM,
		(uintptr_t)sobol_qrng_directions, n_dimensions*n_directions, sizeof(unsigned));

	unsigned *cnt_array = calloc(ntasks, sizeof(unsigned));
	STARPU_ASSERT(cnt_array);
	starpu_data_handle_t cnt_array_handle;
	starpu_vector_data_register(&cnt_array_handle, STARPU_MAIN_RAM, (uintptr_t)cnt_array, ntasks, sizeof(unsigned));

	/* Use a write-through policy : when the data is modified on an
	 * accelerator, we know that it will only be modified once and be
	 * accessed by the CPU later on */
	starpu_data_set_wt_mask(cnt_array_handle, (1<<0));

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = ntasks
	};

	starpu_data_partition(cnt_array_handle, &f);

	double start;
	double end;

	start = starpu_timing_now();

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &pi_cl;

		STARPU_ASSERT(starpu_data_get_sub_data(cnt_array_handle, 1, i));

		task->handles[0] = sobol_qrng_direction_handle;
		task->handles[1] = starpu_data_get_sub_data(cnt_array_handle, 1, i);

		ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_task_wait_for_all();

	/* Get the cnt_array back in main memory */
	starpu_data_unpartition(cnt_array_handle, STARPU_MAIN_RAM);
	starpu_data_unregister(cnt_array_handle);
	starpu_data_unregister(sobol_qrng_direction_handle);

	/* Count the total number of entries */
	unsigned long total_cnt = 0;
	for (i = 0; i < ntasks; i++)
		total_cnt += cnt_array[i];

	end = starpu_timing_now();

	double timing = end - start;

	unsigned long total_shot_cnt = ntasks * nshot_per_task;

	/* Total surface : Pi * r^ 2 = Pi*1^2, total square surface : 2^2 = 4, probability to impact the disk: pi/4 */
	FPRINTF(stderr, "Pi approximation : %f (%lu / %lu)\n", ((TYPE)total_cnt*4)/(total_shot_cnt), total_cnt, total_shot_cnt);
	FPRINTF(stderr, "Total time : %f ms\n", timing/1000.0);
	FPRINTF(stderr, "Speed : %f GShot/s\n", total_shot_cnt/(1e3*timing));

	if (!getenv("STARPU_SSILENT")) starpu_codelet_display_stats(&pi_cl);

	starpu_shutdown();

	return 0;
}
