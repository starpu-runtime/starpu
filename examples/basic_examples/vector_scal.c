/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2013  Université de Bordeaux 1
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
 * This example demonstrates how to use StarPU to scale an array by a factor.
 * It shows how to manipulate data with StarPU's data management library.
 *  1- how to declare a piece of data to StarPU (starpu_vector_data_register)
 *  2- how to describe which data are accessed by a task (task->handles[0])
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 */

#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define	NX	204800
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)


int main(int argc, char **argv)
{
	int * A,*B,*C,*D,*E;

	putenv("STARPU_LIMIT_CPU_MEM=130");

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);

	/* register a disk */
	unsigned dd = starpu_disk_register(&write_on_file, (void *) "/tmp/", 1024*1024*200);

	/* allocate two memory spaces */
	starpu_malloc((void **)&A, NX*sizeof(int));
	starpu_malloc((void **)&E, NX*sizeof(int));

	if (ret == -ENODEV) goto enodev;

	FPRINTF(stderr, "Test of disk memory \n");
	int j;
	for(j = 0; j < NX; ++j)
	{
		A[j] = j;
		E[j] = -j;
	}

	/* Tell StaPU to associate the "vector" vector with the "vector_handle"
	 * identifier. When a task needs to access a piece of data, it should
	 * refer to the handle that is associated to it.
	 * In the case of the "vector" data interface:
	 *  - the first argument of the registration method is a pointer to the
	 *    handle that should describe the data
	 *  - the second argument is the memory node where the data (ie. "vector")
	 *    resides initially: 0 stands for an address in main memory, as
	 *    opposed to an adress on a GPU for instance.
	 *  - the third argument is the adress of the vector in RAM
	 *  - the fourth argument is the number of elements in the vector
	 *  - the fifth argument is the size of each element.
	 */
	starpu_data_handle_t vector_handleA, vector_handleB, vector_handleC, vector_handleD, vector_handleE;

	starpu_vector_data_register(&vector_handleA, 0, (uintptr_t)A, NX, sizeof(int));
	starpu_vector_data_register(&vector_handleB, -1, (uintptr_t) NULL, NX, sizeof(int));	
	starpu_vector_data_register(&vector_handleC, -1, (uintptr_t) NULL, NX, sizeof(int));
	starpu_vector_data_register(&vector_handleD, -1, (uintptr_t) NULL, NX, sizeof(int));
	starpu_vector_data_register(&vector_handleE, 0, (uintptr_t)E, NX, sizeof(int));

	starpu_data_cpy(vector_handleB, vector_handleA, 1, NULL, NULL);
	starpu_data_cpy(vector_handleC, vector_handleB, 1, NULL, NULL);
	starpu_data_cpy(vector_handleD, vector_handleC, 1, NULL, NULL);
	starpu_data_cpy(vector_handleE, vector_handleD, 1, NULL, NULL);

	/* StarPU does not need to manipulate the array anymore so we can stop
 	 * monitoring it */

	starpu_data_unregister(vector_handleA);
	starpu_data_unregister(vector_handleB);
	starpu_data_unregister(vector_handleC);
	starpu_data_unregister(vector_handleD);
	starpu_data_unregister(vector_handleE);

	starpu_disk_unregister(dd);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	int try = 1;
	for (j = 0; j < NX; ++j)
		if (A[j] != E[j])
		{
			printf("fail A %d != E %d \n", A[j], E[j]);
			try = 0;
		}

	if(try)
		FPRINTF(stderr, "TEST SUCCESS\n");
	else
		FPRINTF(stderr, "TEST FAIL\n");
	return (try ? EXIT_SUCCESS : EXIT_FAILURE);

enodev:
	return 77;
}
