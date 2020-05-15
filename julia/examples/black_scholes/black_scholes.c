/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Mael Keryell
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
#include <stdlib.h>
#include <stdio.h>
#include <starpu.h>
#include <math.h>
#include "../includes/sorting.h"



void cpu_black_scholes(void **, void *);
void gpu_black_scholes(void **, void *);

static struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_black_scholes},
	.cuda_funcs = {gpu_black_scholes},
	.nbuffers = 7,
	.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_W}
};

void black_scholes_with_starpu(double *S, double *K, double *R, double *T, double *sig, double *call_res, double *put_res, unsigned nbr_data, unsigned nslices)
{
	starpu_data_handle_t S_handle, K_handle, R_handle, T_handle, SIG_handle, CRES_handle, PRES_handle;
	

	starpu_vector_data_register(&S_handle, STARPU_MAIN_RAM, (uintptr_t)S, nbr_data, sizeof(double));	
	starpu_vector_data_register(&K_handle, STARPU_MAIN_RAM, (uintptr_t)K, nbr_data, sizeof(double));
	starpu_vector_data_register(&R_handle, STARPU_MAIN_RAM, (uintptr_t)R, nbr_data, sizeof(double));
	starpu_vector_data_register(&T_handle, STARPU_MAIN_RAM, (uintptr_t)T, nbr_data, sizeof(double));
	starpu_vector_data_register(&SIG_handle, STARPU_MAIN_RAM, (uintptr_t)sig, nbr_data, sizeof(double));
	starpu_vector_data_register(&CRES_handle, STARPU_MAIN_RAM, (uintptr_t)call_res, nbr_data, sizeof(double));
	starpu_vector_data_register(&PRES_handle, STARPU_MAIN_RAM, (uintptr_t)put_res, nbr_data, sizeof(double));

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = nslices
	};
	/* printf("%f %f\n", nslices, nbr_data); */

	starpu_data_partition(S_handle, &f);
	starpu_data_partition(K_handle, &f);
	starpu_data_partition(R_handle, &f);
	starpu_data_partition(T_handle, &f);
	starpu_data_partition(SIG_handle, &f);
	starpu_data_partition(CRES_handle, &f);
	starpu_data_partition(PRES_handle, &f);
	
	unsigned taskid;

	for (taskid = 0; taskid < nslices; taskid++){

		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;
		task->handles[0] = starpu_data_get_sub_data(S_handle, 1, taskid);
		task->handles[1] = starpu_data_get_sub_data(K_handle, 1, taskid);
		task->handles[2] = starpu_data_get_sub_data(R_handle, 1, taskid);
		task->handles[3] = starpu_data_get_sub_data(T_handle, 1, taskid);
		task->handles[4] = starpu_data_get_sub_data(SIG_handle, 1, taskid);
		task->handles[5] = starpu_data_get_sub_data(CRES_handle, 1, taskid);
		task->handles[6] = starpu_data_get_sub_data(PRES_handle, 1, taskid);
		
		starpu_task_submit(task);

	}

	starpu_task_wait_for_all();

	starpu_data_unpartition(S_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(K_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(R_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(T_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(SIG_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(CRES_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(PRES_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(S_handle);
	starpu_data_unregister(K_handle);
	starpu_data_unregister(R_handle);
	starpu_data_unregister(T_handle);
	starpu_data_unregister(SIG_handle);
	starpu_data_unregister(CRES_handle);
	starpu_data_unregister(PRES_handle);
	


}

static void init_S(double *S, unsigned nbr_data)
{
	unsigned i;
	for (i = 0; i < nbr_data; i++){
		S[i] = 100. * rand() / (double) RAND_MAX;
	}
}

static void init_K(double *K, unsigned nbr_data)
{
	unsigned i;
	for (i = 0; i < nbr_data; i++){
		K[i] = 100. * rand() / (double) RAND_MAX;
	}
}

static void init_R(double *R, unsigned nbr_data)
{
	unsigned i;
	for (i = 0; i < nbr_data; i++){
		R[i] = rand() / (double) RAND_MAX;
	}
}

static void init_T(double *T, unsigned nbr_data)
{
	unsigned i;
	for (i = 0; i < nbr_data; i++){
		T[i] = 10. * rand() / (double) RAND_MAX;
	}
}

static void init_sig(double *sig, unsigned nbr_data)
{
	unsigned i;
	for (i = 0; i < nbr_data; i++){
		sig[i] = 10. * rand() / (double) RAND_MAX;
	}
}


double median_time(unsigned nbr_data, unsigned nslices, unsigned nbr_tests)
{
	double exec_times[nbr_tests];
	
	double *S = malloc(nbr_data * sizeof(double));
	double *K = malloc(nbr_data * sizeof(double));
	double *R = malloc(nbr_data * sizeof(double));
	double *T = malloc(nbr_data * sizeof(double));
	double *sig = malloc(nbr_data * sizeof(double));

	double *call_res = calloc(nbr_data, sizeof(double));
	double *put_res = calloc(nbr_data, sizeof(double));

	double start, stop;
	unsigned i;
	for (i = 0; i < nbr_tests; i++){

		init_S(S,nbr_data);
		init_K(K,nbr_data);
		init_R(R,nbr_data);
		init_T(T,nbr_data);
		init_sig(sig,nbr_data);

		/* S[0] = 100.; */
		/* K[0] = 100.; */
		/* R[0] = 0.05; */
		/* T[0] = 1.0; */
		/* sig[0] = 0.2; */
		
		start = starpu_timing_now();
		black_scholes_with_starpu(S, K, R, T, sig, call_res, put_res, nbr_data, nslices);
		stop = starpu_timing_now();
	
		exec_times[i] = (stop - start) / 1.e6;
	}

	/* printf("%f %f\n", call_res[0], put_res[0]); */

	free(S);
	free(K);
	free(R);
	free(T);
	free(sig);
	free(call_res);
	free(put_res);

	quicksort(exec_times, 0, nbr_tests - 1);

	
	return exec_times[nbr_tests/2];
}
	


void display_times(unsigned start_nbr, unsigned step_nbr, unsigned stop_nbr, unsigned nslices, unsigned nbr_tests)
{
	FILE *myfile;

	myfile = fopen("DAT/black_scholes_c_times.dat", "w");

	unsigned nbr_data;

	for (nbr_data = start_nbr; nbr_data <= stop_nbr; nbr_data += step_nbr){
		double t = median_time(nbr_data, nslices, nbr_tests);
		printf("nbr_data:\n%u\nTime:\n%f\n", nbr_data, t);
		fprintf(myfile, "%f\n", t);
	}
	fclose(myfile);
}

int main(int argc, char *argv[])
{
	if (argc != 6){
		printf("Usage: %s start_nbr step_nbr stop_nbr nslices nbr_tests\n", argv[0]);
		return 1;
	}

	if (starpu_init(NULL) != EXIT_SUCCESS){
		fprintf(stderr, "ERROR\n");
		return 77;
	}

	unsigned start_nbr = (unsigned) atoi(argv[1]);
	unsigned step_nbr = (unsigned) atoi(argv[2]);
	unsigned stop_nbr = (unsigned) atoi(argv[3]);
	unsigned nslices = (unsigned) atoi(argv[4]);
	unsigned nbr_tests = (unsigned) atoi(argv[5]);

	srand(time(NULL));

	display_times(start_nbr, step_nbr, stop_nbr, nslices, nbr_tests);

	starpu_shutdown();

	return 0;
}
		
