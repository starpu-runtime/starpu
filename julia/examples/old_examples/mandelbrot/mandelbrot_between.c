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
#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>
#include <stdint.h>
#include "../includes/display.h"

void mandelbrot(void **, void *);
void CUDA_mandelbrot(void **, void *);
void test(void **, void *); /* Function used to test on my matrix, in the cpu_test_with_generated.c file. */

struct Params
{
	float cr;
	float ci;
	unsigned taskx;
	unsigned tasky;
	unsigned width;
	unsigned height;
};



struct starpu_codelet cl =
{
	/* .cpu_funcs = {test}, */ 
	.cpu_funcs = {mandelbrot},
	.cuda_funcs = {CUDA_mandelbrot},

	//ARRAY PAR
	.nbuffers = 3,
	.modes = {STARPU_RW, STARPU_R, STARPU_R}
	//

	//STRUCT PAR
	/* .nbuffers = 1, */
	/* .modes = {STARPU_RW} */
	//
};


void mandelbrot_with_starpu(int64_t *pixels, float cr, float ci, unsigned width, unsigned height, unsigned nslicesx, unsigned nslicesy, double *params)
{
	//ARRAY PAR
	starpu_data_handle_t p_handle, par_handle, v_handle;
	//

	//STRUCT PAR
	/* starpu_data_handle_t p_handle; */
	//

	starpu_matrix_data_register(&p_handle, STARPU_MAIN_RAM, (uintptr_t)pixels, width, width, height, sizeof(int64_t));
	//ARRAY PAR
	starpu_matrix_data_register(&par_handle, STARPU_MAIN_RAM, (uintptr_t)params, 5, 5, 1, sizeof(double));
	//

	struct starpu_data_filter vert =
	{
		/* .filter_func = starpu_matrix_filter_block, */
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nslicesy
	};

	struct starpu_data_filter horiz =
	{
		.filter_func = starpu_matrix_filter_block,
		/* .filter_func = starpu_matrix_filter_vertical_block, */
		.nchildren = nslicesx
	};

	starpu_data_map_filters(p_handle, 2, &vert, &horiz);

	unsigned taskx, tasky;

	//ARRAY PAR
	int64_t *V = malloc(2 * nslicesx * nslicesy * sizeof(int64_t));
	//

	//STRUCT PAR
	/* struct Params *parameters = malloc(nslicesx * nslicesy * sizeof(struct Params)); */
	//

	for (tasky = 0; tasky < nslicesy; tasky++){
		for (taskx = 0; taskx < nslicesx; taskx++){
			struct starpu_task *task = starpu_task_create();

			//ARRAY PAR			
			V[2 * (taskx + nslicesx * tasky)] = taskx + 1;
			V[2 * (taskx + nslicesx * tasky) + 1] = tasky + 1;
			starpu_vector_data_register(&v_handle, STARPU_MAIN_RAM, (uintptr_t)&(V[2 * (taskx + nslicesx * tasky)]), 2, sizeof(int64_t));
			//
			

			/* printf("Pre-Task%u_%u\n", taskx, tasky); */

			task->cl = &cl;

			
			task->handles[0] = starpu_data_get_sub_data(p_handle, 2, tasky, taskx);
			
			//ARRAY PAR
			task->handles[1] = par_handle;
			task->handles[2] = v_handle;
			//

			//STRUCT PAR
			/* struct Params param = {cr, ci, taskx, tasky, width, height}; */

			

			/* parameters[taskx + tasky * nslicesx] = param; */

			/* task->cl_arg = (parameters + taskx + tasky * nslicesx); */
			/* task->cl_arg_size = sizeof(struct Params); */
			//


			starpu_task_submit(task);

			//ARRAY PAR
			starpu_data_unregister_submit(v_handle);
			//
		}
	}
	starpu_task_wait_for_all();

	starpu_data_unpartition(p_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(p_handle);

	//STRUCT PAR
	/* free(parameters); */
	//

	//ARRAY PAR
	starpu_data_unregister(par_handle);
	/* starpu_data_unregister(v_handle); */
	free(V);
	//
}

void init_zero(int64_t * pixels, unsigned width, unsigned height)
{
	unsigned i,j;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			pixels[j + i*width] = 0;
		}
	}
}

void sort(double *arr, unsigned nbr_tests)
{
	unsigned j;
	
	int is_sort = 0;
	
	while (!is_sort){

		is_sort = 1;
		
		for (j = 0; j < nbr_tests - 1; j++){
			if (arr[j] > arr[j+1]){
				is_sort = 0;
				double tmp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = tmp;
			}
		}
	}
}
double median_time(float cr, float ci, unsigned width, unsigned height, unsigned nslicesx, unsigned nslicesy, unsigned nbr_tests)
{
	int64_t *Pixels = malloc(width*height*sizeof(int64_t));
	double *params = malloc(4*sizeof(double));
	
	double max_iterations = (width/2) * 0.049715909 * log10(width * 0.25296875);

	params[0] = (double) cr;
	params[1] = (double) ci;
	params[2] = (double) width;
	params[3] = (double) height;
	params[4] = (double) max_iterations;
	unsigned i;

	double exec_times[nbr_tests];

	double start, stop, exec_t;
	for (i = 0; i < nbr_tests; i++){
		init_zero(Pixels, width, height);
		
		start = starpu_timing_now(); // starpu_timing_now() gives the time in microseconds.
		mandelbrot_with_starpu(Pixels, cr, ci, width, height, nslicesx, nslicesy, params);
		stop = starpu_timing_now();
		
		exec_t = (stop-start)/1.e6;
		exec_times[i] = exec_t;
	}

	
	char filename[34];
	sprintf(filename, "PPM/mandelbrottest%d.ppm", width);
	printf("%s\n", filename);

	/* Due to Julia registering matrices differently in memory, we need to transpose the matrix we get from the Julia generated kernels */

	mandelbrot_graph_transpose(filename, Pixels, width, height);


	/* mandelbrot_graph_transpose("PPM/mandelbrottest.ppm", Pixels, width, height); */

	free(Pixels);
	free(params);

	sort(exec_times, nbr_tests);

	return exec_times[nbr_tests/2];	
}


void display_times(float cr, float ci, unsigned start_dim, unsigned step_dim, unsigned stop_dim, unsigned nslices, unsigned nbr_tests)
{
	
	unsigned dim;

	FILE *myfile;
	myfile = fopen("DAT/mandelbrot_c_array_times.dat", "w");

	for (dim = start_dim; dim <= stop_dim; dim += step_dim){
		
		double t = median_time(cr, ci, dim, dim, nslices, nslices, nbr_tests);
		
		printf("w = %u ; h = %u ; t = %f\n", dim, dim, t);
		
		fprintf(myfile, "%f\n", t);
		}
	
	fclose(myfile);
}

int main(int argc, char **argv)
{

	if (argc != 8){
		printf("Usage: %s cr ci start_dim step_dim stop_dim nslices(must divide dims) nbr_tests\n", argv[0]);
		return 1;
	}
	if (starpu_init(NULL) != EXIT_SUCCESS){
		fprintf(stderr, "ERROR\n");
		return 77;
	}


	
	float cr = (float) atof(argv[1]);
	float ci = (float) atof(argv[2]);
	unsigned start_dim = (unsigned) atoi(argv[3]);
	unsigned step_dim = (unsigned) atoi(argv[4]);	
	unsigned stop_dim = (unsigned) atoi(argv[5]);
	unsigned nslices = (unsigned) atoi(argv[6]);
	unsigned nbr_tests = (unsigned) atoi(argv[7]);

	display_times(cr, ci, start_dim, step_dim, stop_dim, nslices, nbr_tests);



	starpu_shutdown();


	return 0;
}
