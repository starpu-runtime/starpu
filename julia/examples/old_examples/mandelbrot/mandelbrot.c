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
#include "../display.h"

void cpu_mandelbrot(void **, void *);
void gpu_mandelbrot(void **, void *);

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
	.cpu_funcs = {cpu_mandelbrot},
	.cuda_funcs = {gpu_mandelbrot},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};


void mandelbrot_with_starpu(int *pixels, float cr, float ci, unsigned width, unsigned height, unsigned nslicesx, unsigned nslicesy)
{
	starpu_data_handle_t p_handle;

	starpu_matrix_data_register(&p_handle, STARPU_MAIN_RAM, (uintptr_t)pixels, width, width, height, sizeof(int));

	struct starpu_data_filter vert =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nslicesy
	};

	struct starpu_data_filter horiz =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = nslicesx
	};

	starpu_data_map_filters(p_handle, 2, &vert, &horiz);

	unsigned taskx, tasky;

	struct Params *params = malloc(nslicesx*nslicesy*sizeof(struct Params));

	for (taskx = 0; taskx < nslicesx; taskx++){
		for (tasky = 0; tasky < nslicesy; tasky++){
			struct starpu_task *task = starpu_task_create();
			
			task->cl = &cl;
			task->handles[0] = starpu_data_get_sub_data(p_handle, 2, tasky, taskx);
			struct Params param = {cr, ci, taskx, tasky, width, height};

			params[taskx + tasky*nslicesx] = param;

			task->cl_arg = (params + taskx + tasky * nslicesx);
			task->cl_arg_size = sizeof(struct Params);
			
			starpu_task_submit(task);
		}
	}
	starpu_task_wait_for_all();

	starpu_data_unpartition(p_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(p_handle);

	free(params);
}

void init_zero(int * pixels, unsigned width, unsigned height)
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
	int *Pixels = malloc(width*height*sizeof(int));
	
	unsigned i;

	double exec_times[nbr_tests];

	double start, stop, exec_t;
	for (i = 0; i < nbr_tests; i++){
		init_zero(Pixels, width, height);
		
		start = starpu_timing_now(); // starpu_timing_now() gives the time in microseconds.
		mandelbrot_with_starpu(Pixels, cr, ci, width, height, nslicesx, nslicesy);
		stop = starpu_timing_now();
		
		exec_t = (stop-start)/1.e6;
		exec_times[i] = exec_t;
	}
	char filename[30];
	sprintf(filename, "PPM/mandelbrot%d.ppm", width);
	printf("%s\n", filename);

	mandelbrot_graph(filename, Pixels, width, height);

	free(Pixels);

	sort(exec_times, nbr_tests);

	return exec_times[nbr_tests/2];	
}

void fluctuation_time(float cr, float ci, unsigned width, unsigned height, unsigned nslicesx, unsigned nslicesy, unsigned nbr_tests, double *exec_times)
{
	int *Pixels = malloc(width*height*sizeof(int));
	
	unsigned i;

	double start, stop, exec_t;
	for (i = 0; i < nbr_tests; i++){
		init_zero(Pixels, width, height);
		
		start = starpu_timing_now(); // starpu_timing_now() gives the time in microseconds.
		mandelbrot_with_starpu(Pixels, cr, ci, width, height, nslicesx, nslicesy);
		stop = starpu_timing_now();
		
		exec_t = (stop-start)/1.e6;
		exec_times[i] = exec_t;

		/* char filename[33]; */
		/* sprintf(filename, "../PPM/mandelbrot%d.ppm", i + 1); */
		/* printf("%s\n", filename); */
		/* mandelbrot_graph(filename, Pixels, width, height); */
	}


	free(Pixels);



	
}


void display_times(float cr, float ci, unsigned start_dim, unsigned step_dim, unsigned stop_dim, unsigned nslices, unsigned nbr_tests)
{
	
	unsigned dim;

	FILE *myfile;
	myfile = fopen("DAT/mandelbrot_c_struct_times.dat", "w");

	for (dim = start_dim; dim <= stop_dim; dim += step_dim){
		printf("Dimension: %u...\n", dim);
		double t = median_time(cr, ci, dim, dim, nslices, nslices, nbr_tests);
		
		printf("w = %u ; h = %u ; t = %f\n", dim, dim, t);
		
		fprintf(myfile, "%f\n", t);
		}
	
	fclose(myfile);
}

void display_fluctuations(float cr, float ci, unsigned start_dim, unsigned step_dim, unsigned stop_dim, unsigned nslices, unsigned nbr_tests)
{
	
	unsigned dim;

	FILE *myfile;
	myfile = fopen("DAT/mandelbrot_c_fluctuation.dat", "w");

	double *exec_times = malloc(nbr_tests * sizeof(double));
	fluctuation_time(cr, ci, start_dim, start_dim, nslices, nslices, nbr_tests, exec_times);
		
	/* printf("w = %u ; h = %u ; t = %f\n", dim, dim, t); */
	unsigned i;
	for (i = 0; i < nbr_tests; i++){
		printf("test %u: %f seconds\n", i, exec_times[i]);
		fprintf(myfile, "%u %f\n", i, exec_times[i]);
	}
	
	fclose(myfile);
	free(exec_times);
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
	
	
	/* display_fluctuations(cr, ci, start_dim, step_dim, stop_dim, nslices, nbr_tests); */


	starpu_shutdown();


	return 0;
}
