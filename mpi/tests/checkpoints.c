/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi.h>
#include "helper.h"

#define ARRAY_SIZE 12

int nb_nodes;
int me;

int backup_of(int _me)
{
	if (_me==0)
		return 1;
	else
		return 0;
	return (_me+1)%nb_nodes;
}

int pseudotest_checkpoint_template_register(int argc, char* argv[])
{
	int mpi_init;
	starpu_data_handle_t             h;
	starpu_data_handle_t             h_array[ARRAY_SIZE];
	starpu_mpi_checkpoint_template_t cp_template1, cp_template2;
	int                              val = 42;
	int                              val2 = 1234;

	int array[ARRAY_SIZE];
	int ret;

	//init array
	for (int i=0 ; i<ARRAY_SIZE ; i++)
	{
		array[i] = i*1111+42;
	}

	for (int i=0 ; i<ARRAY_SIZE ; i++)
	{
		h_array[i] = NULL;
	}
	FPRINTF(stderr, "Go\n");

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	FPRINTF_MPI(stderr, "Init ok - my rnk %d - size %d\n", me, nb_nodes);

	starpu_variable_data_register(&h, STARPU_MAIN_RAM, (uintptr_t)&val2, sizeof(int));
	starpu_mpi_data_register(h, 56, 0);
	fprintf(stderr, "&h: %p, h:%p\n", &h, h);
	for (int i=0 ; i<ARRAY_SIZE ; i++)
	{
		starpu_variable_data_register(&h_array[i], STARPU_MAIN_RAM, (uintptr_t)&array[i], sizeof(int));
		starpu_mpi_data_register(h_array[i], 42+i, 1); //42 to 54
	}

	starpu_mpi_checkpoint_template_register(&cp_template1, 123486, 0,
	                                        STARPU_VALUE, &val, sizeof(int), 84, backup_of,
	                                        STARPU_R, h, 1,
	                                        0);

	FPRINTF(stderr, "registered!\n");
	starpu_mpi_checkpoint_template_print(cp_template1);

	starpu_mpi_checkpoint_template_create(&cp_template2, 98765, 0);
	starpu_mpi_checkpoint_template_add_entry(&cp_template2, STARPU_R, h, 1);
	starpu_mpi_checkpoint_template_add_entry(&cp_template2, STARPU_VALUE, &val, sizeof(int), 84, backup_of);
	starpu_mpi_checkpoint_template_freeze(&cp_template2);

	FPRINTF(stderr, "registered 2!\n");
	starpu_mpi_checkpoint_template_print(cp_template1);

	starpu_shutdown();
	return 0;
}

int test_checkpoint_submit(int argc, char* argv[])
{
	int ret;
	starpu_data_handle_t handle0, handle1;
	starpu_mpi_checkpoint_template_t cp_template;
	int val0 = 0;
	int val1 = 0;
	int stage = 10;

	FPRINTF(stderr, "Go\n");

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_size(MPI_COMM_WORLD, &nb_nodes);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &me);

	stage+=me;

	FPRINTF_MPI(stderr, "Init ok - my rnk %d - size %d\n", me, nb_nodes);

	starpu_variable_data_register(&handle0, STARPU_MAIN_RAM, (uintptr_t)&val0, sizeof(int));
	starpu_mpi_data_register(handle0, 100, 0);

	starpu_variable_data_register(&handle1, STARPU_MAIN_RAM, (uintptr_t)&val1, sizeof(int));
	starpu_mpi_data_register(handle1, 200, 1);

	FPRINTF_MPI(stderr, "Registering\n");
	starpu_mpi_checkpoint_template_register(&cp_template, 321, 0,
						STARPU_R, handle0, 1,
						STARPU_R, handle1, 0,
						STARPU_VALUE, &stage, sizeof(int), 300, &backup_of,
						STARPU_VALUE, &stage, sizeof(int), 301, &backup_of,
						STARPU_VALUE, &stage, sizeof(int), 302, &backup_of,
						0);
	FPRINTF_MPI(stderr, "Registered\n");

	starpu_mpi_checkpoint_template_print(cp_template);

	switch (me)
	{
		case 0:
			val0 = 42;
			break;
		case 1:
			val1 = 1000;
			break;
	}
	FPRINTF_MPI(stderr, "Submitting\n");
	starpu_mpi_checkpoint_template_submit(cp_template,0);

	FPRINTF_MPI(stderr, "Submitted\n");

	usleep(150000);
	stage++;
	fprintf(stderr, "\n\n");
	usleep(150000);

	starpu_data_acquire(handle0, STARPU_RW);
	if (me==0)
		val0 *= 2;
	starpu_data_release(handle0);

	starpu_data_acquire(handle1, STARPU_RW);
	if (me==1)
		val1*=2;
	starpu_data_release(handle1);

	FPRINTF_MPI(stderr, "Submitting\n");
	starpu_mpi_checkpoint_template_submit(cp_template, 0);

	FPRINTF_MPI(stderr, "Submitted\n");

	sleep(2);
	fprintf(stderr, "\n\n");
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	FPRINTF_MPI(stderr, "Bye!\n");
	starpu_mpi_shutdown();

	return 0;
}

int main(int argc, char* argv[])
{
	//pseudotest_checkpoint_template_register(argc, argv);
	test_checkpoint_submit(argc, argv);
	return 0;
}
