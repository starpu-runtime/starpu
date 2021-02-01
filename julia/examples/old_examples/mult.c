/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2018       Alexis Juven
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
 * This example shows a simple implementation of a blocked matrix
 * multiplication. Note that this is NOT intended to be an efficient
 * implementation of sgemm! In this example, we show:
 *  - how to declare dense matrices (starpu_matrix_data_register)
 *  - how to manipulate matrices within codelets (eg. descr[0].blas.ld)
 *  - how to use filters to partition the matrices into blocks
 *    (starpu_data_partition and starpu_data_map_filters)
 *  - how to unpartition data (starpu_data_unpartition) and how to stop
 *    monitoring data (starpu_data_unregister)
 *  - how to manipulate subsets of data (starpu_data_get_sub_data)
 *  - how to construct an autocalibrated performance model (starpu_perfmodel)
 *  - how to submit asynchronous tasks
 */

#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <signal.h>

#include <starpu.h>



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





void gpu_mult(void **, void *);
void cpu_mult(void **, void *);





static struct starpu_perfmodel model =
{
		.type = STARPU_HISTORY_BASED,
		.symbol = "history_perf"
};

static struct starpu_codelet cl =
{
		.cpu_funcs = {cpu_mult},
		.cpu_funcs_name = {"cpu_mult"},
		.cuda_funcs = {gpu_mult},
		.nbuffers = 3,
		.modes = {STARPU_R, STARPU_R, STARPU_W},
		.model = &model
};






void multiply_with_starpu(float *A, float *B, float *C,  unsigned xdim,  unsigned ydim,  unsigned zdim, unsigned nslicesx, unsigned nslicesy)
{
	starpu_data_handle_t A_handle, B_handle, C_handle;


	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A,
			ydim, ydim, zdim, sizeof(float));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B,
			zdim, zdim, xdim, sizeof(float));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C,
			ydim, ydim, xdim, sizeof(float));


	struct starpu_data_filter vert =
	{
			.filter_func = starpu_matrix_filter_vertical_block,
			.nchildren = nslicesx
	};

	struct starpu_data_filter horiz =
	{
			.filter_func = starpu_matrix_filter_block,
			.nchildren = nslicesy
	};


	starpu_data_partition(B_handle, &vert);
	starpu_data_partition(A_handle, &horiz);
	starpu_data_map_filters(C_handle, 2, &vert, &horiz);

	unsigned taskx, tasky;

	for (taskx = 0; taskx < nslicesx; taskx++){
		for (tasky = 0; tasky < nslicesy; tasky++){

			struct starpu_task *task = starpu_task_create();

			task->cl = &cl;
			task->handles[0] = starpu_data_get_sub_data(A_handle, 1, tasky);
			task->handles[1] = starpu_data_get_sub_data(B_handle, 1, taskx);
			task->handles[2] = starpu_data_get_sub_data(C_handle, 2, taskx, tasky);

			starpu_task_submit(task);

		}
	}

	starpu_task_wait_for_all();


	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

}



void init_rand(float * m, unsigned width, unsigned height)
{
	unsigned i,j;

	for (j = 0 ; j < height ; j++){
		for (i = 0 ; i < width ; i++){
			m[j+i*height] = (float)(starpu_drand48());
		}
	}
}


void init_zero(float * m, unsigned width, unsigned height)
{
	memset(m, 0, sizeof(float) * width * height);
}



void sort(unsigned int size, double t[])
{
	unsigned int j;

	int is_sort = 0;

	while(!is_sort){

		is_sort = 1;

		for (j = 0 ; j < size - 1 ; j++){

			if (t[j] > t[j+1]){
				double tmp = t[j];
				t[j] = t[j+1];
				t[j+1] = tmp;
				is_sort = 0;
			}
		}
	}


}


double median_time(unsigned nb_test, unsigned xdim, unsigned ydim, unsigned zdim, unsigned nsclicesx, unsigned nsclicesy)
{
	unsigned i;

	float * A = (float *) malloc(zdim*ydim*sizeof(float));
	float * B = (float *) malloc(xdim*zdim*sizeof(float));
	float * C = (float *) malloc(xdim*ydim*sizeof(float));

	double exec_times[nb_test];

	for (i = 0 ; i < nb_test ; i++){

		double start, stop, exec_t;

		init_rand(A, zdim, ydim);
		init_rand(B, xdim, zdim);
		init_zero(C, xdim, ydim);

		start = starpu_timing_now();
		multiply_with_starpu(A, B, C, xdim, ydim, zdim, nsclicesx, nsclicesy);
		stop = starpu_timing_now();

		exec_t = (stop - start)/1.e6;
		exec_times[i] = exec_t;
	}

	sort(nb_test, exec_times);

	free(A);
	free(B);
	free(C);

	return exec_times[nb_test/2];
}


void display_times(unsigned start_dim, unsigned step_dim, unsigned stop_dim, unsigned nb_tests, unsigned nsclicesx, unsigned nsclicesy)
{
	unsigned dim;

	for (dim = start_dim ; dim <= stop_dim ; dim += step_dim){
		double t = median_time(nb_tests, dim, dim, dim, nsclicesx, nsclicesy);
		printf("%u ; %f\n", dim, t);
	}

}


int main(int argc, char * argv[])
{

	if (argc != 7){
		printf("Usage : %s start_dim step_dim stop_dim nb_tests nsclicesx nsclicesy\n", argv[0]);
		return 1;
	}


	if (starpu_init(NULL) != EXIT_SUCCESS){
		fprintf(stderr, "ERROR\n");
		return 77;
	}

	unsigned start_dim = (unsigned) atoi(argv[1]);
	unsigned step_dim = (unsigned) atoi(argv[2]);
	unsigned stop_dim = (unsigned) atoi(argv[3]);
	unsigned nb_tests = (unsigned) atoi(argv[4]);
	unsigned nsclicesx = (unsigned) atoi(argv[5]);
	unsigned nsclicesy = (unsigned) atoi(argv[6]);

	display_times(start_dim, step_dim, stop_dim, nb_tests, nsclicesx, nsclicesy);

	starpu_shutdown();

	return 0;
}

