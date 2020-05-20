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
#include "../includes/display.h"
#include "../includes/sorting.h"

struct Param {
	unsigned taskx;
	double epsilon;
};

void cpu_nbody(void **, void *);
void cpu_nbody2(void **, void *);

void gpu_nbody(void **, void *);
void gpu_nbody2(void **, void *);

static struct starpu_perfmodel model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "history_perf"
};

static struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_nbody},
	.cuda_funcs = {gpu_nbody},
	.nbuffers = 3,
	.modes = {STARPU_RW, STARPU_RW, STARPU_RW},
	.model = &model
};


static struct starpu_codelet cl2 =
{
	.cpu_funcs = {cpu_nbody2},
	.cuda_funcs = {gpu_nbody2},
	.nbuffers = 3,
	.modes = {STARPU_RW, STARPU_RW, STARPU_R},
	.model = &model
};

void nbody_with_starpu(double *positions, double *velocities, double *accelerations, double *masses, unsigned nbr_planets, unsigned nbr_simulations, unsigned nslices)
{



	starpu_data_handle_t P_handle, V_handle, A_handle, M_handle;

       
	starpu_matrix_data_register(&P_handle, STARPU_MAIN_RAM, (uintptr_t)positions, nbr_planets, nbr_planets, 2, sizeof(double));
	starpu_matrix_data_register(&V_handle, STARPU_MAIN_RAM, (uintptr_t)velocities, nbr_planets, nbr_planets, 2, sizeof(double));
	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)accelerations, nbr_planets, nbr_planets, 2, sizeof(double));
	starpu_matrix_data_register(&M_handle, STARPU_MAIN_RAM, (uintptr_t)masses, nbr_planets, nbr_planets, 1, sizeof(double));
	
	struct starpu_data_filter horiz = 
		{
			.filter_func = starpu_matrix_filter_block,
			.nchildren = nslices
		};
	
	unsigned i;
		
	for (i = 0; i < nbr_simulations; i++){
		
		/* printf("Simulation %d: \n", i); */
		
		starpu_data_partition(A_handle, &horiz);
		
		struct Param *params = malloc(nslices * sizeof(struct Param));
		
		
		unsigned epsilon = 2.5e8;
		
		unsigned task1, task2;
		
		
		
		for (task1 = 0; task1 < nslices; task1++){
			
			struct starpu_task *task = starpu_task_create();
			
			struct Param param = {task1, epsilon};
			
			params[task1] = param;
			
			task->cl = &cl;
			task->handles[0] = P_handle;
			task->handles[1] = starpu_data_get_sub_data(A_handle, 1, task1);
			task->handles[2] = M_handle;
			
			task->cl_arg = &(params[task1]);
			task->cl_arg_size = sizeof(struct Param);
			
			starpu_task_submit(task);
		}
		starpu_task_wait_for_all();

////////////////////////

		starpu_data_partition(P_handle, &horiz);
		starpu_data_partition(V_handle, &horiz);
		
		for (task2 = 0; task2 < nslices; task2++){
			
			struct starpu_task *task = starpu_task_create();
			
			struct Param param = {task1, epsilon};
			
			params[task2] = param;
			
			task->cl = &cl2;
			task->handles[0] = starpu_data_get_sub_data(P_handle, 1, task2);
			task->handles[1] = starpu_data_get_sub_data(V_handle, 1, task2);
			task->handles[2] = starpu_data_get_sub_data(A_handle, 1, task2);
			
			task->cl_arg = &(params[task2]);
			task->cl_arg_size = sizeof(struct Param);
			
			starpu_task_submit(task);
		}
		
		starpu_task_wait_for_all();
		
		starpu_data_unpartition(P_handle, STARPU_MAIN_RAM);
		starpu_data_unpartition(V_handle, STARPU_MAIN_RAM);
		starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);

		/* char filename[41]; */
	
		/* sprintf(filename, "../PPM/nbody%d_%d.ppm", nbr_planets, i + 1); */
	
		/* nbody_graph(filename, positions, nbr_planets, 1000, 1000, -4e8, 4e8); */

	}
	
		starpu_data_unregister(P_handle);
		starpu_data_unregister(V_handle);
		starpu_data_unregister(A_handle);
		starpu_data_unregister(M_handle);


}

void init_positions(double *positions, unsigned nbr_planets)
{
	unsigned i;
	double qiX, qiY;
	for (i = 0; i < nbr_planets; i++){
		double angle = ((RAND_MAX - rand()) / (double) (RAND_MAX)) * 2.0 * M_PI;
		double distToCenter = ((RAND_MAX - rand()) / (double) (RAND_MAX)) * 1.0e8 + 1.0e8;

		qiX = cos(angle) * distToCenter;
		qiY = sin(angle) * distToCenter;

		positions[i] = qiX;
		positions[i + nbr_planets] = qiY;
	}
}

void init_velocities(double *positions, double *velocities, unsigned nbr_planets)
{
	unsigned i;
	for (i = 0; i < nbr_planets; i++){
		double viX = positions[i + nbr_planets] * 4.0e-6;
		double viY = -positions[i] * 4.0e-6;
		velocities[i] = viX;
		velocities[i + nbr_planets] = viY;
	}
}

void init_masses(double *masses, unsigned nbr_planets)
{
	unsigned i;
	for (i = 0; i < nbr_planets; i++){
		double mi = (rand() / (double) RAND_MAX) * 5e25;
		masses[i] = mi;
	}
}


	

double median_times(unsigned nbr_planets, unsigned nbr_simulations, unsigned nslices, unsigned nbr_tests)
{
	double exec_times[nbr_tests];

	double *positions = malloc(nbr_planets * 2 * sizeof(double));
	double *velocities = malloc(nbr_planets * 2 * sizeof(double));
	double *accelerations = calloc(nbr_planets * 2, sizeof(double));
	double *masses = malloc(nbr_planets * sizeof(double));

	init_positions(positions, nbr_planets);
	init_velocities(positions, velocities, nbr_planets);
	init_masses(masses, nbr_planets);


	/* unsigned i; */

	/* for (i = 0; i < nbr_planets; i++){ */
	/* 	accelerations[i] = i; */
	/* } */

	
	
	/* nbody_graph("PPM/bug0.ppm", positions, nbr_planets, 1000, 1000, -4e8, 4e8); */

	/* unsigned k; */
       
	/* for (k = 1; k <= nbr_simulations; k++){ */
	
	double start, stop, exec_t;
	unsigned i;
	
	for (i = 0; i < nbr_tests; i++){

		start = starpu_timing_now();

		nbody_with_starpu(positions, velocities, accelerations, masses, nbr_planets, nbr_simulations, nslices);
		
		stop = starpu_timing_now();

		exec_t = (stop - start) / 1.e6;

		exec_times[i] = exec_t;
	}
	/* printf("\n\nSIMULATION %d:\n\n", k); */
	
	
	/* char filename[23]; */
	
	/* sprintf(filename, "PPM/bug%d.ppm", k); */
	
	/* nbody_graph(filename, positions, nbr_planets, 1000, 1000, -4e8, 4e8); */
	/* } */
	
	free(positions);
	free(velocities);
	free(accelerations);
	free(masses);

	quicksort(exec_times, 0, nbr_tests - 1);

	return exec_times[nbr_tests / 2];
}



void display_times(unsigned start_nbr, unsigned step_nbr, unsigned stop_nbr, unsigned nbr_simulations, unsigned nslices, unsigned nbr_tests)
{

	FILE *myfile;
	
	myfile = fopen("DAT/nbody_c_struct_times.dat", "w");

	unsigned nbr_planets;
	
	for (nbr_planets = start_nbr; nbr_planets <= stop_nbr; nbr_planets += step_nbr){
		double t = median_times(nbr_planets, nbr_simulations, nslices, nbr_tests);
		printf("STRUCT: %u planets: %f seconds\n", nbr_planets, t);
		fprintf(myfile, "%f\n", t);
	}
	fclose(myfile);
}

int main(int argc, char * argv[])
{

	if (argc != 7){
		printf("Usage: %s start_nbr step_nbr stop_nbr nbr_simulations nslices nbr_tests\n", argv[0]);
		return 1;
	}


	if (starpu_init(NULL) != EXIT_SUCCESS){
		fprintf(stderr, "ERROR\n");
		return 77;
	}


	unsigned start_nbr = (unsigned) atoi(argv[1]);
	unsigned step_nbr = (unsigned) atoi(argv[2]);
	unsigned stop_nbr = (unsigned) atoi(argv[3]);
	unsigned nbr_simulations = (unsigned) atoi(argv[4]);
	unsigned nslices = (unsigned) atoi(argv[5]);
	unsigned nbr_tests = (unsigned) atoi(argv[6]);

	srand(time(NULL));
	
	display_times(start_nbr, step_nbr, stop_nbr, nbr_simulations, nslices, nbr_tests);

	starpu_shutdown();
	
	return 0;
}
